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

////cuda v10.2 ��thrust��
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
#include "openMVG/matching/metric.hpp"
#include "openMVG/numeric/eigen_alias_definition.hpp"
#include "openMVG/stl/dynamic_bitset.hpp"

using namespace Eigen;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> RowMatrixXf;

extern "C"
float computeEuclideanDistance
(
	const unsigned char * descriptionData2,
	const unsigned char * descriptionData1,
	size_t size
);

//const Eigen::VectorXf primary_projection = primary_hash_projection_ * descriptor;
//����primary_hash_projection_�Ѿ���GPU����
//extern "C"
//float primaryMulDescriptor
//(
//	float * descriptor,
//	float * result,
//	size_t size
//);

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
			// ������ת�����ȣ�ֻ�ı�洢��ʽ�����ı����ߴ�
			void rToC(const float *a, int ha, int wa, float *b)// �������鼰������������
			{
				int i;
				float *temp = (float *)malloc(sizeof(float) * ha * wa);// ��������ȴ洢����ʱ����

				for (i = 0; i < ha * wa; i++)         // �ҳ������ȵĵ� i ��λ�ö�Ӧ������������и�ֵ
					temp[i] = a[i / ha + i % ha * wa];
				for (i = 0; i < ha * wa; i++)         // ����ԭ����
					b[i] = temp[i];
				free(temp);
				return;
			}

			// ������ת������
			void cToR(const float *a, int ha, int wa, float *b)
			{
				int i;
				float *temp = (float *)malloc(sizeof(float) * ha * wa);

				for (i = 0; i < ha * wa; i++)         // �ҳ������ȵĵ� i ��λ�ö�Ӧ������������и�ֵ
					temp[i] = a[i / wa + i % wa * ha];
				for (i = 0; i < ha * wa; i++)
					b[i] = temp[i];
				free(temp);
				return;
			}
		};

		class cublas_Mul {
		public:
			// ������Ծ����ά��
			int const Arows = 128;
			int const Acols = 128;
			int const Brows = 128;
			int const Bcols = 1;

			int Mul(Eigen::MatrixXf primary_hash_projection_, Eigen::VectorXf descriptor, float *h_C)
			{
				// ����״̬����
				cublasStatus_t status;

				// �� �ڴ� ��Ϊ��Ҫ����ľ��󿪱ٿռ�
				float *h_A = (float*)malloc(Arows * Acols * sizeof(float));
				Eigen::Map<RowMatrixXf>(h_A, Arows, Acols) = primary_hash_projection_;
				float *h_B = (float*)malloc(Acols * Bcols * sizeof(float));
				Eigen::Map<RowMatrixXf>(h_B, Brows, Bcols) = descriptor;

				//// �� �ڴ� ��Ϊ��Ҫ����������ľ��󿪱ٿռ�
				//float *h_C = (float*)malloc(Arows*Bcols * sizeof(float));

				/*
				** GPU ����������
				*/

				// ��������ʼ�� CUBLAS �����
				cublasHandle_t handle;
				status = cublasCreate(&handle);

				if (status != CUBLAS_STATUS_SUCCESS)
				{
					if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
						cout << "CUBLAS ����ʵ��������" << endl;
					}
					getchar();
					return EXIT_FAILURE;
				}

				float *d_A, *d_B, *d_C;
				// �� �Դ� ��Ϊ��Ҫ����ľ��󿪱ٿռ�
				cudaMalloc(
					(void**)&d_A,    // ָ�򿪱ٵĿռ��ָ��
					Arows* Acols * sizeof(float)    //����Ҫ���ٿռ���ֽ���
				);
				cudaMalloc(
					(void**)&d_B,
					Brows * Bcols * sizeof(float)
				);

				// �� �Դ� ��Ϊ��Ҫ����������ľ��󿪱ٿռ�
				cudaMalloc(
					(void**)&d_C,
					Arows * Bcols * sizeof(float)
				);

				// ���������ݴ��ݽ� �Դ� ���Ѿ����ٺ��˵Ŀռ�
				cublasSetVector(
					Arows * Acols,    // Ҫ�����Դ��Ԫ�ظ���
					sizeof(float),    // ÿ��Ԫ�ش�С
					h_A,    // ��������ʼ��ַ
					1,    // ����Ԫ��֮��Ĵ洢���
					d_A,    // GPU ����ʼ��ַ
					1    // ����Ԫ��֮��Ĵ洢���
				);
				cublasSetVector(
					Brows * Bcols,
					sizeof(float),
					h_B,
					1,
					d_B,
					1
				);

				// ͬ������
				cudaThreadSynchronize();

				// ���ݽ�������˺����еĲ��������庬����ο������ֲᡣ
				float a = 1; float b = 0;
				// ������ˡ��ú�����Ȼ���������������������
				cublasSgemm(
					handle,    // blas �����
					CUBLAS_OP_T,    // ���� A ���Բ���
					CUBLAS_OP_T,    // ���� B ���Բ���
					Arows,    // A, C ������
					Bcols,    // B, C ������
					Acols,    // A �������� B ������
					&a,    // ����ʽ�� �� ֵ
					d_A,    // A ���Դ��еĵ�ַ
					Acols,    // lda
					d_B,    // B ���Դ��еĵ�ַ
					Bcols,    // ldb
					&b,    // ����ʽ�� �� ֵ
					d_C,    // C ���Դ��еĵ�ַ(�������)
					Arows    // ldc
				);

				// ͬ������
				cudaThreadSynchronize();

				// �� �Դ� ��ȡ���������� �ڴ���ȥ
				cublasGetVector(
					Arows*Bcols,    //  Ҫȡ��Ԫ�صĸ���
					sizeof(float),    // ÿ��Ԫ�ش�С
					d_C,    // GPU ����ʼ��ַ
					1,    // ����Ԫ��֮��Ĵ洢���
					h_C,    // ��������ʼ��ַ
					1    // ����Ԫ��֮��Ĵ洢���
				);


				// �����ʹ�ù����ڴ�
				free(h_A);
				free(h_B);
				//free(h_C);
				cudaFree(d_A);
				cudaFree(d_B);
				cudaFree(d_C);

				// �ͷ� CUBLAS �����
				cublasDestroy(handle);
				
				return EXIT_SUCCESS;
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
				return descriptions.template cast<float>().colwise().mean();
			}
			
			//my GPU parallels here
			template <typename MatrixT>
			HashedDescriptions CreateHashedDescriptions
			(
				const MatrixT & descriptions,
				const Eigen::VectorXf & zero_mean_descriptor
			) const
			{
				// Steps:
				//   1) Compute hash code and hash buckets (based on the zero_mean_descriptor).
				//   2) Construct buckets.

				HashedDescriptions hashed_descriptions;
				if (descriptions.rows() == 0) {
					return hashed_descriptions;
				}

				// Create hash codes for each description.
				{
					// Allocate space for hash codes.
					const typename MatrixT::Index nbDescriptions = descriptions.rows();
					hashed_descriptions.hashed_desc.resize(nbDescriptions);
					Eigen::VectorXf descriptor(descriptions.cols());
					for (int i = 0; i < nbDescriptions; ++i)
					{
						// Allocate space for each bucket id.
						hashed_descriptions.hashed_desc[i].bucket_ids.resize(nb_bucket_groups_);

						// Compute hash code.
						auto& hash_code = hashed_descriptions.hashed_desc[i].hash_code;
						hash_code = stl::dynamic_bitset(descriptions.cols());
						descriptor = descriptions.row(i).template cast<float>();
						descriptor -= zero_mean_descriptor;
						//const Eigen::VectorXf primary_projection = primary_hash_projection_ * descriptor;
						
						float *primaryMulDescriptor = (float*)malloc(nb_hash_code_  * sizeof(float));
						cublas_Mul myCublas_Mul;
						myCublas_Mul.Mul(primary_hash_projection_, descriptor, primaryMulDescriptor);

						for (int j = 0; j < nb_hash_code_; ++j)
						{
							hash_code[j] = primaryMulDescriptor[j] > 0;
						}

						// Determine the bucket index for each group.
						Eigen::VectorXf secondary_projection;
						for (int j = 0; j < nb_bucket_groups_; ++j)
						{
							uint16_t bucket_id = 0;
							secondary_projection = secondary_hash_projection_[j] * descriptor;

							for (int k = 0; k < nb_bits_per_bucket_; ++k)
							{
								bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
							}
							hashed_descriptions.hashed_desc[i].bucket_ids[j] = bucket_id;
						}
					}
				}
				// Build the Buckets
				{
					hashed_descriptions.buckets.resize(nb_bucket_groups_);
					for (int i = 0; i < nb_bucket_groups_; ++i)
					{
						hashed_descriptions.buckets[i].resize(nb_buckets_per_group_);

						// Add the descriptor ID to the proper bucket group and id.
						for (int j = 0; j < hashed_descriptions.hashed_desc.size(); ++j)
						{
							const uint16_t bucket_id = hashed_descriptions.hashed_desc[j].bucket_ids[i];
							hashed_descriptions.buckets[i][bucket_id].push_back(j);
						}
					}
				}
				return hashed_descriptions;
			}

			//template <typename MatrixT>
			//HashedDescriptions CreateHashedDescriptions
			//(
			//	const MatrixT & descriptions,
			//	const Eigen::VectorXf & zero_mean_descriptor
			//) const
			//{
			//	// Steps:
			//	//   1) Compute hash code and hash buckets (based on the zero_mean_descriptor).
			//	//   2) Construct buckets.

			//	HashedDescriptions hashed_descriptions;
			//	if (descriptions.rows() == 0) {
			//		return hashed_descriptions;
			//	}

			//	// Create hash codes for each description.
			//	{
			//		// Allocate space for hash codes.
			//		const typename MatrixT::Index nbDescriptions = descriptions.rows();
			//		hashed_descriptions.hashed_desc.resize(nbDescriptions);
			//		Eigen::VectorXf descriptor(descriptions.cols());
			//		for (int i = 0; i < nbDescriptions; ++i)
			//		{
			//			// Allocate space for each bucket id.
			//			hashed_descriptions.hashed_desc[i].bucket_ids.resize(nb_bucket_groups_);

			//			// Compute hash code.
			//			auto& hash_code = hashed_descriptions.hashed_desc[i].hash_code;
			//			hash_code = stl::dynamic_bitset(descriptions.cols());
			//			descriptor = descriptions.row(i).template cast<float>();
			//			descriptor -= zero_mean_descriptor;
			//			const Eigen::VectorXf primary_projection = primary_hash_projection_ * descriptor;
			//			for (int j = 0; j < nb_hash_code_; ++j)
			//			{
			//				hash_code[j] = primary_projection(j) > 0;
			//			}

			//			// Determine the bucket index for each group.
			//			Eigen::VectorXf secondary_projection;
			//			for (int j = 0; j < nb_bucket_groups_; ++j)
			//			{
			//				uint16_t bucket_id = 0;
			//				secondary_projection = secondary_hash_projection_[j] * descriptor;

			//				for (int k = 0; k < nb_bits_per_bucket_; ++k)
			//				{
			//					bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
			//				}
			//				hashed_descriptions.hashed_desc[i].bucket_ids[j] = bucket_id;
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
			//	return hashed_descriptions;
			//}

			float* hash_gen
			(
				//��ʽB������
				int mat_I_cols_count,
				//��ʽA���������� ��ʽB��������ΪdescriptionDimension
				int descriptionDimension,
				//��ʽA
				float *primary_hash_projection_data_device,
				//��ʽB
				const float *descriptions_GPU
			)
			{
				/*dim3 threads(1, 1);
				dim3 grid(1, 1);*/
				// ��������ʼ�� CUBLAS �����
				cublasHandle_t handle;
				int status = cublasCreate(&handle);
				if (status != CUBLAS_STATUS_SUCCESS)
				{
					if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
						std::cout << "CUBLAS ����ʵ��������" << std::endl;
					}
					getchar();

				}
				// ͬ������
				cudaThreadSynchronize();
				// ���ݽ�������˺����еĲ��������庬����ο������ֲᡣ
				float a = 1; float b = 0;
				// ������ˡ��ú�����Ȼ���������������������
				
				float *result;
				cudaMalloc((void **)&result, sizeof(float) * mat_I_cols_count * descriptionDimension);

				status = cublasSgemm
				(
					//A(primary_hash_proejection)
					//B(description_Mat)
					//C(result)
					handle, //blas�����
					CUBLAS_OP_N, //����A��ת��
					CUBLAS_OP_N, //����B��ת��
					descriptionDimension, //����A��C������,Ҳ�����������
					mat_I_cols_count, //����B��C��������Ҳ�����������
					descriptionDimension, //����A����������B������
					&a,  //alpha��ֵ
					primary_hash_projection_data_device, //�����A
					descriptionDimension, //A��leading dimension,����Ϊ����������
					descriptions_GPU, //�Ҿ��� B
					descriptionDimension, //����B��leading dimension,����Ϊ����������
					&b,             //beta��ֵ
					result, //�������C
					descriptionDimension//�������C��leading dimension,����Ϊ����������
				);
				cublasDestroy(handle);
				return result;
			}

			float* determine_buket_index_for_each_group
			(
				
				//��ʽA
				float *secondary_hash_projection_j,
				//��ʽB
				const float *descriptions_GPU,
				//��ʽA������  10
				int secondary_rows,
				//��ʽA������ 128
				int secondary_cols,
				//��ʽB������ 
				int descrptions_cols
				//��ʽB����������descriptionDimension
			)
			{
				/*dim3 threads(1, 1);
				dim3 grid(1, 1);*/
				// ��������ʼ�� CUBLAS �����
				cublasHandle_t handle;
				int status = cublasCreate(&handle);
				if (status != CUBLAS_STATUS_SUCCESS)
				{
					if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
						std::cout << "CUBLAS ����ʵ��������" << std::endl;
					}
					getchar();

				}
				// ͬ������
				cudaThreadSynchronize();
				// ���ݽ�������˺����еĲ��������庬����ο������ֲᡣ
				float a = 1; float b = 0;
				// ������ˡ��ú�����Ȼ���������������������

				float *result;
				cudaMalloc((void **)&result, sizeof(float) * secondary_rows * descrptions_cols);

				status = cublasSgemm
				(
					//A(primary_hash_proejection)
					//B(description_Mat)
					//C(result)
					handle, //blas�����
					CUBLAS_OP_N, //����A��ת��
					CUBLAS_OP_N, //����B��ת��
					secondary_rows, //����A��C������,Ҳ�����������
					descrptions_cols, //����B��C��������Ҳ�����������
					//computeMatches::descriptionDimension, //����A����������B������
					secondary_cols, //A������ B������
					&a,  //alpha��ֵ
					secondary_hash_projection_j, //�����A
					secondary_rows, //A��leading dimension,����Ϊ����������
					descriptions_GPU, //�Ҿ��� B
					computeMatches::descriptionDimension, //����B��leading dimension,����Ϊ����������
					&b,             //beta��ֵ
					result, //�������C
					secondary_rows//�������C��leading dimension
				);
				cublasDestroy(handle);
				return result;
			}


			// my GPU parallel here
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
				for (int i = 0; i < hashed_descriptions1.hashed_desc.size(); ++i)
				{
					candidate_descriptors.clear();
					num_descriptors_with_hamming_distance.setZero();
					candidate_euclidean_distances.clear();

					const auto& hashed_desc = hashed_descriptions1.hashed_desc[i];

					// Accumulate all descriptors in each bucket group that are in the same
					// bucket id as the query descriptor.
					for (int j = 0; j < nb_bucket_groups_; ++j)
					{
						const uint16_t bucket_id = hashed_desc.bucket_ids[j];
						for (const auto& feature_id : hashed_descriptions2.buckets[j][bucket_id])
						{
							candidate_descriptors.emplace_back(feature_id);
							used_descriptor[feature_id] = false;
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

					// Compute the euclidean distance of the k descriptors with the best hamming
					// distance.
					candidate_euclidean_distances.reserve(kNumTopCandidates);
					for (int j = 0; j < candidate_hamming_distances.cols() &&
						(candidate_euclidean_distances.size() < kNumTopCandidates); ++j)
					{
						for (int k = 0; k < num_descriptors_with_hamming_distance(j) &&
							(candidate_euclidean_distances.size() < kNumTopCandidates); ++k)
						{
							const int candidate_id = candidate_hamming_distances(k, j);
							const DistanceType distance = metric(
								descriptions2.row(candidate_id).data(),
								descriptions1.row(i).data(),
								descriptions1.cols());

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
			}
			

			//// Matches two collection of hashed descriptions with a fast matching scheme
			//// based on the hash codes previously generated.
			//template <typename MatrixT, typename DistanceType>
			//void Match_HashedDescriptions
			//(
			//	const HashedDescriptions& hashed_descriptions1,
			//	const MatrixT & descriptions1,
			//	const HashedDescriptions& hashed_descriptions2,
			//	const MatrixT & descriptions2,
			//	IndMatches * pvec_indices,
			//	std::vector<DistanceType> * pvec_distances,
			//	const int NN = 2
			//) const
			//{
			//	using MetricT = L2<typename MatrixT::Scalar>;
			//	MetricT metric;

			//	static const int kNumTopCandidates = 10;

			//	// Preallocate the candidate descriptors container.
			//	std::vector<int> candidate_descriptors;
			//	candidate_descriptors.reserve(hashed_descriptions2.hashed_desc.size());

			//	// Preallocated hamming distances. Each column indicates the hamming distance
			//	// and the rows collect the descriptor ids with that
			//	// distance. num_descriptors_with_hamming_distance keeps track of how many
			//	// descriptors have that distance.
			//	Eigen::MatrixXi candidate_hamming_distances(
			//		hashed_descriptions2.hashed_desc.size(), nb_hash_code_ + 1);
			//	Eigen::VectorXi num_descriptors_with_hamming_distance(nb_hash_code_ + 1);

			//	// Preallocate the container for keeping euclidean distances.
			//	std::vector<std::pair<DistanceType, int>> candidate_euclidean_distances;
			//	candidate_euclidean_distances.reserve(kNumTopCandidates);

			//	// A preallocated vector to determine if we have already used a particular
			//	// feature for matching (i.e., prevents duplicates).
			//	std::vector<bool> used_descriptor(hashed_descriptions2.hashed_desc.size());

			//	using HammingMetricType = matching::Hamming<stl::dynamic_bitset::BlockType>;
			//	static const HammingMetricType metricH = {};
			//	for (int i = 0; i < hashed_descriptions1.hashed_desc.size(); ++i)
			//	{
			//		candidate_descriptors.clear();
			//		num_descriptors_with_hamming_distance.setZero();
			//		candidate_euclidean_distances.clear();

			//		const auto& hashed_desc = hashed_descriptions1.hashed_desc[i];

			//		// Accumulate all descriptors in each bucket group that are in the same
			//		// bucket id as the query descriptor.
			//		for (int j = 0; j < nb_bucket_groups_; ++j)
			//		{
			//			const uint16_t bucket_id = hashed_desc.bucket_ids[j];
			//			for (const auto& feature_id : hashed_descriptions2.buckets[j][bucket_id])
			//			{
			//				candidate_descriptors.emplace_back(feature_id);
			//				used_descriptor[feature_id] = false;
			//			}
			//		}

			//		// Skip matching this descriptor if there are not at least NN candidates.
			//		if (candidate_descriptors.size() <= NN)
			//		{
			//			continue;
			//		}

			//		// Compute the hamming distance of all candidates based on the comp hash
			//		// code. Put the descriptors into buckets corresponding to their hamming
			//		// distance.
			//		for (const int candidate_id : candidate_descriptors)
			//		{
			//			if (!used_descriptor[candidate_id]) // avoid selecting the same candidate multiple times
			//			{
			//				used_descriptor[candidate_id] = true;

			//				const HammingMetricType::ResultType hamming_distance = metricH(
			//					hashed_desc.hash_code.data(),
			//					hashed_descriptions2.hashed_desc[candidate_id].hash_code.data(),
			//					hashed_desc.hash_code.num_blocks());
			//				candidate_hamming_distances(
			//					num_descriptors_with_hamming_distance(hamming_distance)++,
			//					hamming_distance) = candidate_id;
			//			}
			//		}

			//		// Compute the euclidean distance of the k descriptors with the best hamming
			//		// distance.
			//		candidate_euclidean_distances.reserve(kNumTopCandidates);
			//		for (int j = 0; j < candidate_hamming_distances.cols() &&
			//			(candidate_euclidean_distances.size() < kNumTopCandidates); ++j)
			//		{
			//			for (int k = 0; k < num_descriptors_with_hamming_distance(j) &&
			//				(candidate_euclidean_distances.size() < kNumTopCandidates); ++k)
			//			{
			//				const int candidate_id = candidate_hamming_distances(k, j);
			//				const DistanceType distance = metric(
			//					descriptions2.row(candidate_id).data(),
			//					descriptions1.row(i).data(),
			//					descriptions1.cols());

			//				candidate_euclidean_distances.emplace_back(distance, candidate_id);
			//			}
			//		}

			//		// Assert that each query is having at least NN retrieved neighbors
			//		if (candidate_euclidean_distances.size() >= NN)
			//		{
			//			// Find the top NN candidates based on euclidean distance.
			//			std::partial_sort(candidate_euclidean_distances.begin(),
			//				candidate_euclidean_distances.begin() + NN,
			//				candidate_euclidean_distances.end());
			//			// save resulting neighbors
			//			for (int l = 0; l < NN; ++l)
			//			{
			//				pvec_distances->emplace_back(candidate_euclidean_distances[l].first);
			//				pvec_indices->emplace_back(IndMatch(i, candidate_euclidean_distances[l].second));
			//			}
			//		}
			//		//else -> too few candidates... (save no one)
			//	}
			//}
		
			// Primary hashing function.
			Eigen::MatrixXf primary_hash_projection_;

			// Secondary hashing function.
			std::vector<Eigen::MatrixXf> secondary_hash_projection_;
		};

	}  // namespace matchingGPU
}  // namespace openMVG

#endif // OPENMVG_MATCHING_CASCADE_HASHER_GPU_HPP
   
   
  