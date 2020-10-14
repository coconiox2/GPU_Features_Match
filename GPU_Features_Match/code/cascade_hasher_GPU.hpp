#pragma once
#ifndef OPENMVG_MATCHING_CASCADE_HASHER_GPU_HPP
#define OPENMVG_MATCHING_CASCADE_HASHER_GPU_HPP
#include "computeMatchesCU.h"
//CUDA V 10.2

#include <algorithm>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

////cuda v10.2 的thrust库
//#include <thrust/device_vector.h>
//#include <thrust/host_vector.h>
//#include <thrust/generate.h>
//#include <thrust/sort.h>
//#include <thrust/copy.h>

#include <cstdlib>

#include <stdio.h>
#include <cmath>
#include <ctime>
#include <fstream>  
#include <iostream>
#include <random>
#include <utility>
#include <map> 
#include <vector>
#include <string>
#include <sstream>  

#include "openMVG/matching/indMatch.hpp"
//#include "metric.hpp"
#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/stl/dynamic_bitset.hpp"

extern "C"
float computeEuclideanDistance
(
	const unsigned char * descriptionData2,
	const unsigned char * descriptionData1,
	size_t size
);

using namespace computeMatches;
namespace openMVG {
	namespace matching {
		
		

		// This hasher will hash descriptors with a two-step hashing system:
		// 1. it generates a hash code,
		// 2. it determines which buckets the descriptors belong to.
		// Retrieval step is fast since:
		// - only descriptors in the same bucket are likely to be good matches.
		//   - 1. a hamming distance is used for fast neighbor candidate retrieval
		//   - 2. the L2 distance is computed only a reduced selection of approximate neighbor
		//
		// Implementation is based on the paper [1].
		// If you use this matcher, please cite the paper.
		class RTOC {
		public:
			// 行优先转列优先，只改变存储方式，不改变矩阵尺寸
			void rToC(const float *a, int ha, int wa, float *b)// 输入数组及其行数、列数
			{
				int i;
				float *temp = (float *)malloc(sizeof(float) * ha * wa);// 存放列优先存储的临时数组

				for (i = 0; i < ha * wa; i++)         // 找出列优先的第 i 个位置对应行优先坐标进行赋值
					temp[i] = a[i / ha + i % ha * wa];
				for (i = 0; i < ha * wa; i++)         // 覆盖原数组
					b[i] = temp[i];
				free(temp);
				return;
			}

			// 列优先转行优先
			void cToR(const float *a, int ha, int wa, float *b)
			{
				int i;
				float *temp = (float *)malloc(sizeof(float) * ha * wa);

				for (i = 0; i < ha * wa; i++)         // 找出行优先的第 i 个位置对应列优先坐标进行赋值
					temp[i] = a[i / wa + i % wa * ha];
				for (i = 0; i < ha * wa; i++)
					b[i] = temp[i];
				free(temp);
				return;
			}
		};
		
		class CascadeHasherGPU {
		public:

			// The number of bucket bits.
			int nb_bits_per_bucket_;
			// The number of dimensions of the Hash code.
			int nb_hash_code_;
			// The number of bucket groups.
			int nb_bucket_groups_;
			// The number of buckets in each group.
			int nb_buckets_per_group_;

		
			CascadeHasherGPU() = default;

			// Creates the hashing projections (cascade of two level of hash codes)
			bool Init
			(
				const uint8_t nb_hash_code = 128,
				//L functions
				const uint8_t nb_bucket_groups = 6,
				//m bits hashcode
				const uint8_t nb_bits_per_bucket = 10,
				const unsigned random_seed = std::mt19937::default_seed)
			{
				nb_bucket_groups_ = nb_bucket_groups;
				nb_hash_code_ = nb_hash_code;
				nb_bits_per_bucket_ = nb_bits_per_bucket;
				nb_buckets_per_group_ = 1 << nb_bits_per_bucket;

				//
				// Box Muller transform is used in the original paper to get fast random number
				// from a normal distribution with <mean = 0> and <variance = 1>.
				// Here we use C++11 normal distribution random number generator
				std::mt19937 gen(random_seed);
				std::normal_distribution<> d(0, 1);

				primary_hash_projection_.resize(nb_hash_code, nb_hash_code);

				// Initialize primary hash projection.
				for (int i = 0; i < nb_hash_code; ++i)
				{
					for (int j = 0; j < nb_hash_code; ++j)
						primary_hash_projection_(i, j) = d(gen);
				}

				// Initialize secondary hash projection.
				secondary_hash_projection_.resize(nb_bucket_groups);
				for (int i = 0; i < nb_bucket_groups; ++i)
				{
					secondary_hash_projection_[i].resize(nb_bits_per_bucket_,
						nb_hash_code);
					for (int j = 0; j < nb_bits_per_bucket_; ++j)
					{
						for (int k = 0; k < nb_hash_code; ++k)
							secondary_hash_projection_[i](j, k) = d(gen);
					}
				}
				return true;
			}

			template <typename MatrixT>
			Eigen::VectorXf GetZeroMeanDescriptor
			(
				const MatrixT & descriptions
			)
			{
				if (descriptions.rows() == 0) {
					return{};
				}
				return descriptions.template cast<float>().rowwise().mean();
			}
			
			/*
			
			
			//template <typename MatrixT>
			//void CreateHashedDescriptions

			//(
			//	const std::vector<MatrixT> & vec_descriptions,
			//	const Eigen::VectorXf & zero_mean_descriptor,
			//	std::map<IndexT, HashedDescriptions>  & hashed_descriptions
			//) const
			//{
			//	// Steps:
			//	//   1) Compute hash code and hash buckets (based on the zero_mean_descriptor).
			//	//   2) Construct buckets.



			//	//HashedDescriptions hashed_descriptions;
			//	if (descriptions.rows() == 0) {
			//		return hashed_descriptions;
			//	}
			//	//////////////////////////////////////////////////////////////////////////////////////////////
			//	//GPU parallel : Create hash codes for each description.
			//	//
			//	
			//	//host_vec_descriptions = vec_descriptions;

			//	
			//	{
			//		// Allocate space for hash codes.
			//		const typename MatrixT::Index nbDescriptions = descriptions.rows();
			//		hashed_descriptions.hashed_desc.resize(nbDescriptions);


			//		//整合所有的描述符放在一个矩阵里传到device
			//		/*descriptor = descriptions.row(i).template cast<float>();
			//		descriptor -= zero_mean_descriptor;*/
			//		Eigen::MatrixXf primary_projection;
			//		Eigen::MatrixXf descriptionsMat;
			//		descriptionsMat = descriptions.template cast<float>();
			//		for (int k = 0; k < descriptionsMat.rows(); k++) {
			//			descriptionsMat.row(k) -= zero_mean_descriptor;
			//		}


			//		const float *descriptionsMat_data_temp = descriptionsMat.data();
			//		const float *primary_hash_projection_data_temp = primary_hash_projection_.data();
			//		const float *secondary_hash_projection_data_temp = secondary_hash_projection_.data();

			//		float *descriptionsMat_data_temp_1 = (float*)malloc(descriptions.rows()*descriptions.cols() * sizeof(float));
			//		float *primary_hash_projection_data_temp_1 = (float*)malloc(primary_hash_projection_.rows()*primary_hash_projection_.cols() * sizeof(float));
			//		//float *secondary_hash_projection_data_temp_1 = (float )

			//		clock_t start = clock();
			//		//行优先转化为列优先
			//		RTOC sRTOC;
			//		sRTOC.cToR(descriptionsMat_data_temp, descriptionsMat.rows(), descriptionsMat.cols(), descriptionsMat_data_temp_1);
			//		sRTOC.cToR(primary_hash_projection_data_temp, primary_hash_projection_.rows(), primary_hash_projection_.cols(), primary_hash_projection_data_temp_1);
			//		//RTOC::cToR(descriptionsMat_data_temp, descriptionsMat.rows(), descriptionsMat.cols(), descriptionsMat_data_temp_1);
			//		//RTOC::cToR(primary_hash_projection_data_temp, primary_hash_projection_.rows(), primary_hash_projection_.cols(), primary_hash_projection_data_temp_1);

			//		const float *descriptionsMat_data = const_cast<const float*> (descriptionsMat_data_temp_1);
			//		const float *primary_hash_projection_data = const_cast<const float*>(primary_hash_projection_data_temp_1);
			//		//const Eigen::MatrixXf primary_projection = descriptionsMat * primary_hash_projection_;

			//		int descriptions_size = nbDescriptions * (descriptions.cols());
			//		int primary_hash_projection_size = (primary_hash_projection_.rows()) * (primary_hash_projection_.cols());
			//		int result_size = nbDescriptions * (primary_hash_projection_.cols());
			//		//CPU存放结果的指针
			//		float *mat_result_data = (float*)malloc(result_size * sizeof(float));
			//		/*
			//		** GPU 计算矩阵相乘
			//		*/
			//		dim3 threads(1, 1);
			//		dim3 grid(1, 1);
			//		// 创建并初始化 CUBLAS 库对象
			//		cublasHandle_t handle;
			//		int status = cublasCreate(&handle);

			//		if (status != CUBLAS_STATUS_SUCCESS)
			//		{
			//			if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
			//				std::cout << "CUBLAS 对象实例化出错" << std::endl;
			//			}
			//			getchar();

			//		}
			//		//GPU中存放计算矩阵和计算结果的指针
			//		float *descriptionsMat_data_device,
			//			*primary_hash_projection_data_device,
			//			*mat_result_device;
			//		cudaMalloc((void **)&descriptionsMat_data_device, sizeof(float) * descriptions_size);
			//		cudaMalloc((void **)&primary_hash_projection_data_device, sizeof(float) * primary_hash_projection_size);
			//		cudaMalloc((void **)&mat_result_device, sizeof(float) * result_size);

			//		// 将矩阵数据传递进 显存 中已经开辟好了的空间
			//		cudaMemcpy(descriptionsMat_data_device, descriptionsMat_data, sizeof(float) * descriptions_size, cudaMemcpyHostToDevice);
			//		cudaMemcpy(primary_hash_projection_data_device, primary_hash_projection_data, sizeof(float) * primary_hash_projection_size, cudaMemcpyHostToDevice);

			//		// 同步函数
			//		cudaThreadSynchronize();
			//		// 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
			//		float a = 1; float b = 0;
			//		// 矩阵相乘。该函数必然将数组解析成列优先数组
			//		status = cublasSgemm(
			//			handle,    // blas 库对象
			//			CUBLAS_OP_N,    // 矩阵 A 属性参数
			//			CUBLAS_OP_N,    // 矩阵 B 属性参数	
			//			primary_hash_projection_.cols(),    // B, C 的列数, n
			//			descriptions.rows(),    // A, C 的行数 m
			//			descriptions.cols(),    // A 的列数和 B 的行数 k
			//			&a,    // 运算式的 α 值
			//			primary_hash_projection_data_device,    // B 在显存中的地址
			//			primary_hash_projection_.cols(),    // B, C 的列数, n
			//			descriptionsMat_data_device,    // A 在显存中的地址
			//			descriptions.cols(),    // ldb k
			//			&b,    // 运算式的 β 值
			//			mat_result_device,    // C 在显存中的地址(结果矩阵)
			//			primary_hash_projection_.cols()    // B, C 的列数, n
			//		);
			//		cublasDestroy(handle);
			//		// 将在GPU端计算好的结果拷贝回CPU端
			//		//int result_size = nbDescriptions * (primary_hash_projection_.cols());
			//		status = cublasGetMatrix(
			//			nbDescriptions,//m
			//			primary_hash_projection_.cols(),//n
			//			sizeof(*mat_result_data),
			//			mat_result_device,//d_c
			//			nbDescriptions,//m
			//			mat_result_data,//c
			//			nbDescriptions//m
			//		);


			//		for (int i = 0; i < descriptions.rows(); ++i) {
			//			// Allocate space for each bucket id.
			//			hashed_descriptions.hashed_desc[i].bucket_ids.resize(nb_bucket_groups_);
			//			// Compute hash code.
			//			auto& hash_code = hashed_descriptions.hashed_desc[i].hash_code;
			//			hash_code = stl::dynamic_bitset(descriptions.cols());
			//			for (int j = 0; j < nb_hash_code_; ++j)
			//			{
			//				hash_code[j] = mat_result_data[(i*nb_hash_code_ + j)] > 0;
			//			}
			//		}
			//		//计时
			//		clock_t end = clock();
			//		double endtime = (double)(end - start) / CLOCKS_PER_SEC;

			//		//std::cout << "Total time:" << endtime << std::endl;		//s为单位
			//		std::cout << "Task(createHashedDescriptions) cost:" << endtime * 1000 << "ms" << std::endl;	//ms为单位

			//																			// 清理掉使用过的内存

			//		free(mat_result_data);
			//		cudaFree(descriptionsMat_data_device);
			//		cudaFree(primary_hash_projection_data_device);
			//		cudaFree(mat_result_device);

			//		// 释放 CUBLAS 库对象
			//		cublasDestroy(handle);

			//		// Determine the bucket index for each group.
			//		Eigen::MatrixXf secondary_projection;
			//		for (int j = 0; j < nb_bucket_groups_; ++j)
			//		{
			//			uint16_t bucket_id = 0;
			//			secondary_projection = secondary_hash_projection_[j] * descriptionsMat;

			//			//有待测试
			//			for (int h = 0; h < descriptionsMat.rows(); h++) {
			//				for (int k = 0; k < nb_bits_per_bucket_; ++k)
			//				{
			//					bucket_id = (bucket_id << 1) + (secondary_projection(h, k) > 0 ? 1 : 0);
			//				}

			//				hashed_descriptions.hashed_desc[h].bucket_ids[j] = bucket_id;
			//			}
			//		}


			//	}


			//	// Build the Buckets
			//	{
			//		hashed_descriptions.buckets.resize(nb_bucket_groups_);
			//		for (int i = 0; i < nb_bucket_groups_; ++i)
			//		{
			//			hashed_descriptions.buckets[i].resize(nb_buckets_per_group_);

			//			// Add the descriptor ID to the proper bucket group and id.
			//			for (int j = 0; j < hashed_descriptions.hashed_desc.size(); ++j)
			//			{
			//				const uint16_t bucket_id = hashed_descriptions.hashed_desc[j].bucket_ids[i];
			//				hashed_descriptions.buckets[i][bucket_id].push_back(j);
			//			}
			//		}
			//	}
			//	//return hashed_descriptions;
			//}*/
			float* hash_gen
			(
				//右式B的列数
				int mat_I_cols_count,
				//左式A的行数列数 右式B的行数均为descriptionDimension
				int descriptionDimension,
				//左式A
				float *primary_hash_projection_data_device,
				//右式B
				const float *descriptions_GPU
			)
			{
				/*dim3 threads(1, 1);
				dim3 grid(1, 1);*/
				// 创建并初始化 CUBLAS 库对象
				cublasHandle_t handle;
				int status = cublasCreate(&handle);
				if (status != CUBLAS_STATUS_SUCCESS)
				{
					if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
						std::cout << "CUBLAS 对象实例化出错" << std::endl;
					}
					getchar();

				}
				// 同步函数
				cudaThreadSynchronize();
				// 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
				float a = 1; float b = 0;
				// 矩阵相乘。该函数必然将数组解析成列优先数组
				
				float *result;
				cudaMalloc((void **)&result, sizeof(float) * mat_I_cols_count * descriptionDimension);

				status = cublasSgemm
				(
					//A(primary_hash_proejection)
					//B(description_Mat)
					//C(result)
					handle, //blas库对象
					CUBLAS_OP_N, //矩阵A不转置
					CUBLAS_OP_N, //矩阵B不转置
					descriptionDimension, //矩阵A、C的行数,也即结果的行数
					mat_I_cols_count, //矩阵B、C的列数，也即结果的列数
					descriptionDimension, //矩阵A的列数或者B的行数
					&a,  //alpha的值
					primary_hash_projection_data_device, //左矩阵A
					descriptionDimension, //A的leading dimension,以列为主就填行数
					descriptions_GPU, //右矩阵 B
					descriptionDimension, //矩阵B的leading dimension,以列为主就填行数
					&b,             //beta的值
					result, //结果矩阵C
					descriptionDimension//结果矩阵C的leading dimension,以列为主就填行数
				);
				cublasDestroy(handle);
				return result;
			}

			float* determine_buket_index_for_each_group
			(
				
				//左式A
				float *secondary_hash_projection_j,
				//右式B
				const float *descriptions_GPU,
				//左式A的行数  10
				int secondary_rows,
				//左式A的列数 128
				int secondary_cols,
				//右式B的列数 
				int descrptions_cols
				//右式B的行数就是descriptionDimension
			)
			{
				/*dim3 threads(1, 1);
				dim3 grid(1, 1);*/
				// 创建并初始化 CUBLAS 库对象
				cublasHandle_t handle;
				int status = cublasCreate(&handle);
				if (status != CUBLAS_STATUS_SUCCESS)
				{
					if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
						std::cout << "CUBLAS 对象实例化出错" << std::endl;
					}
					getchar();

				}
				// 同步函数
				cudaThreadSynchronize();
				// 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
				float a = 1; float b = 0;
				// 矩阵相乘。该函数必然将数组解析成列优先数组

				float *result;
				cudaMalloc((void **)&result, sizeof(float) * secondary_rows * descrptions_cols);

				status = cublasSgemm
				(
					//A(primary_hash_proejection)
					//B(description_Mat)
					//C(result)
					handle, //blas库对象
					CUBLAS_OP_N, //矩阵A不转置
					CUBLAS_OP_N, //矩阵B不转置
					secondary_rows, //矩阵A、C的行数,也即结果的行数
					descrptions_cols, //矩阵B、C的列数，也即结果的列数
					//computeMatches::descriptionDimension, //矩阵A的列数或者B的行数
					secondary_cols, //A的列数 B的行数
					&a,  //alpha的值
					secondary_hash_projection_j, //左矩阵A
					secondary_rows, //A的leading dimension,以列为主就填行数
					descriptions_GPU, //右矩阵 B
					computeMatches::descriptionDimension, //矩阵B的leading dimension,以列为主就填行数
					&b,             //beta的值
					result, //结果矩阵C
					secondary_rows//结果矩阵C的leading dimension
				);
				cublasDestroy(handle);
				return result;
			}

			// Matches two collection of hashed descriptions with a fast matching scheme
			// based on the hash codes previously generated.
			template <typename MatrixT, typename DistanceType>
			void Match_HashedDescriptions
			(
				const HashedDescriptions& hashed_descriptions1,
				const MatrixT & descriptions1,
				const HashedDescriptions& hashed_descriptions2,
				const MatrixT & descriptions2,
				IndMatches * pvec_indices,
				std::vector<DistanceType> * pvec_distances,
				const int NN = 2
			) const
			{
				using MetricT = L2<typename MatrixT::Scalar>;
				MetricT metric;

				static const int kNumTopCandidates = 10;

				// Preallocate the candidate descriptors container.
				std::vector<int> candidate_descriptors;
				candidate_descriptors.reserve(hashed_descriptions2.hashed_desc.size());

				// Preallocated hamming distances. Each column indicates the hamming distance
				// and the rows collect the descriptor ids with that
				// distance. num_descriptors_with_hamming_distance keeps track of how many
				// descriptors have that distance.
				Eigen::MatrixXi candidate_hamming_distances(
					hashed_descriptions2.hashed_desc.size(), nb_hash_code_ + 1);
				Eigen::VectorXi num_descriptors_with_hamming_distance(nb_hash_code_ + 1);

				// Preallocate the container for keeping euclidean distances.
				std::vector<std::pair<DistanceType, int>> candidate_euclidean_distances;
				candidate_euclidean_distances.reserve(kNumTopCandidates);

				// A preallocated vector to determine if we have already used a particular
				// feature for matching (i.e., prevents duplicates).
				std::vector<bool> used_descriptor(hashed_descriptions2.hashed_desc.size());

				using HammingMetricType = matching::Hamming<stl::dynamic_bitset::BlockType>;
				static const HammingMetricType metricH = {};

				
				//std::cout << std::endl << " - Match_HashedDescriptions_loop - " << std::endl;

				for (int i = 0; i < hashed_descriptions1.hashed_desc.size(); ++i)
				{
					//std::cout << "i: " << i << std::endl;
					candidate_descriptors.clear();
					num_descriptors_with_hamming_distance.setZero();
					candidate_euclidean_distances.clear();

					const auto& hashed_desc = hashed_descriptions1.hashed_desc[i];

					// Accumulate all descriptors in each bucket group that are in the same
					// bucket id as the query descriptor.
					for (int j = 0; j < nb_bucket_groups_; ++j)
					{
						const uint16_t bucket_id = hashed_desc.bucket_ids[j];
						/*if ((j > hashed_descriptions2.buckets.size()-1) 
							|| (bucket_id > hashed_descriptions2.buckets[j].size()-1))
						{
							std::cout << "sss" << std::endl;
						}*/
						
						if (hashed_descriptions2.buckets.size()
							&& hashed_descriptions2.buckets[j].size()
							&& hashed_descriptions2.buckets[j][bucket_id].size())
						{
							for (const auto& feature_id : hashed_descriptions2.buckets[j][bucket_id])
							{
								if (used_descriptor.size() > feature_id + 1 
									|| used_descriptor.size() == feature_id + 1)
								{
									candidate_descriptors.emplace_back(feature_id);
									used_descriptor[feature_id] = false;
								}else
								{
									std::cout << "error in match2descriptions!" << std::endl;
								}
							}
						}
					}

					// Skip matching this descriptor if there are not at least NN candidates.
					if (candidate_descriptors.size() <= NN)
					{
						continue;
					}

					// Compute the hamming distance of all candidates based on the comp hash
					// code. Put the descriptors into buckets corresponding to their hamming
					// distance.
					for (const int candidate_id : candidate_descriptors)
					{
						if (!used_descriptor[candidate_id]) // avoid selecting the same candidate multiple times
						{
							used_descriptor[candidate_id] = true;

							const HammingMetricType::ResultType hamming_distance = metricH(
								hashed_desc.hash_code.data(),
								hashed_descriptions2.hashed_desc[candidate_id].hash_code.data(),
								hashed_desc.hash_code.num_blocks());
							candidate_hamming_distances(
								num_descriptors_with_hamming_distance(hamming_distance)++,
								hamming_distance) = candidate_id;
						}
					}

					//GPU parallel here

					// Compute the euclidean distance of the k descriptors with the best hamming
					// distance.
					candidate_euclidean_distances.reserve(kNumTopCandidates);
					for (int j = 0; j < candidate_hamming_distances.cols() &&
						(candidate_euclidean_distances.size() < kNumTopCandidates); ++j)
					{
						//std::cout << j << std::endl;
						for (int k = 0; k < num_descriptors_with_hamming_distance(j) &&
							(candidate_euclidean_distances.size() < kNumTopCandidates); ++k)
						{
							const int candidate_id = candidate_hamming_distances(k, j);
							//此时的matrix.data()的类型是unsigned char *
							//descriptionsMat = mat_I.template cast<float>();
							if (candidate_id > descriptions2.cols() - 1 || i > descriptions1.cols() - 1) 
							{
								std::cout <<"Error indexing when finding the candidate_id!"<< std::endl;
							}
							const DistanceType distance = computeEuclideanDistance
															(
																descriptions2.col(candidate_id).data(),
																descriptions1.col(i).data(),
																descriptions1.rows()
															);
							/*const DistanceType distance = metric(
								descriptions2.row(candidate_id).data(),
								descriptions1.row(i).data(),
								descriptions1.cols());*/

							candidate_euclidean_distances.emplace_back(distance, candidate_id);
						}
					}

					// Assert that each query is having at least NN retrieved neighbors
					if (candidate_euclidean_distances.size() >= NN)
					{
						// Find the top NN candidates based on euclidean distance.
						std::partial_sort(candidate_euclidean_distances.begin(),
							candidate_euclidean_distances.begin() + NN,
							candidate_euclidean_distances.end());
						// save resulting neighbors
						for (int l = 0; l < NN; ++l)
						{
							pvec_distances->emplace_back(candidate_euclidean_distances[l].first);
							pvec_indices->emplace_back(IndMatch(i, candidate_euclidean_distances[l].second));
						}
					}
					//else -> too few candidates... (save no one)
				}
				//std::cout << "loop over!" << std::endl;
			}

		
			// Primary hashing function.
			Eigen::MatrixXf primary_hash_projection_;

			// Secondary hashing function.
			std::vector<Eigen::MatrixXf> secondary_hash_projection_;
		};

	}  // namespace matchingGPU
}  // namespace openMVG

#endif // OPENMVG_MATCHING_CASCADE_HASHER_GPU_HPP
   
   
  