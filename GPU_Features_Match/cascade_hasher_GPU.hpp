#pragma once
// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef OPENMVG_MATCHING_CASCADE_HASHER_GPU_HPP
#define OPENMVG_MATCHING_CASCADE_HASHER_GPU_HPP

//------------------
//-- Bibliography --
//------------------
//- [1] "Fast and Accurate Image Matching with Cascade Hashing for 3D Reconstruction"
//- Authors: Jian Cheng, Cong Leng, Jiaxiang Wu, Hainan Cui, Hanqing Lu.
//- Date: 2014.
//- Conference: CVPR.
//
// This implementation is based on the Theia library implementation.
//
// Update compare to the initial paper [1] and initial author code:
// - hashing projection is made by using Eigen to use vectorization (Theia)
// - replace the BoxMuller random number generation by C++ 11 random number generation (OpenMVG)
// - this implementation can support various descriptor length and internal type (OpenMVG)
// -  SIFT, SURF, ... all scalar based descriptor
//

// Copyright (C) 2014 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

//CUDA V 10.2

#include <algorithm>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

//cuda v10.2 ��thrust��
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

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

#include "computeMatchesCU.h"
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
		//class cascadeHasher_hash {
		//public:
		//	void Hash
		//	(
		//		//offer openMVG::features & openMVG::descriptor
		//		const std::shared_ptr<sfm::Regions_Provider> & regions_provider,
		//		std::map<IndexT, HashedDescriptions> & hashed_base_,
		//		C_Progress * my_progress_bar
		//	)
		//	{
		//	
		//	}
		//};
		class CascadeHasher {
		public:

			// The number of bucket bits.
			int nb_bits_per_bucket_;
			// The number of dimensions of the Hash code.
			int nb_hash_code_;
			// The number of bucket groups.
			int nb_bucket_groups_;
			// The number of buckets in each group.
			int nb_buckets_per_group_;

		
			CascadeHasher() = default;

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
			static Eigen::VectorXf GetZeroMeanDescriptor
			(
				const MatrixT & descriptions
			)
			{
				if (descriptions.rows() == 0) {
					return{};
				}
				// Compute the ZeroMean descriptor
				return descriptions.template cast<float>().colwise().mean();
			}

			template <typename MatrixT>
			void CreateHashedDescriptions

			(
				const std::vector<MatrixT> & vec_descriptions,
				const Eigen::VectorXf & zero_mean_descriptor,
				std::map<IndexT, HashedDescriptions>  & hashed_descriptions
			) const
			{
				// Steps:
				//   1) Compute hash code and hash buckets (based on the zero_mean_descriptor).
				//   2) Construct buckets.



				//HashedDescriptions hashed_descriptions;
				if (descriptions.rows() == 0) {
					return hashed_descriptions;
				}
				//////////////////////////////////////////////////////////////////////////////////////////////
				//GPU parallel : Create hash codes for each description.
				//
				
				//host_vec_descriptions = vec_descriptions;

				
				{
					// Allocate space for hash codes.
					const typename MatrixT::Index nbDescriptions = descriptions.rows();
					hashed_descriptions.hashed_desc.resize(nbDescriptions);


					//�������е�����������һ�������ﴫ��device
					/*descriptor = descriptions.row(i).template cast<float>();
					descriptor -= zero_mean_descriptor;*/
					Eigen::MatrixXf primary_projection;
					Eigen::MatrixXf descriptionsMat;
					descriptionsMat = descriptions.template cast<float>();
					for (int k = 0; k < descriptionsMat.rows(); k++) {
						descriptionsMat.row(k) -= zero_mean_descriptor;
					}


					const float *descriptionsMat_data_temp = descriptionsMat.data();
					const float *primary_hash_projection_data_temp = primary_hash_projection_.data();
					const float *secondary_hash_projection_data_temp = secondary_hash_projection_.data();

					float *descriptionsMat_data_temp_1 = (float*)malloc(descriptions.rows()*descriptions.cols() * sizeof(float));
					float *primary_hash_projection_data_temp_1 = (float*)malloc(primary_hash_projection_.rows()*primary_hash_projection_.cols() * sizeof(float));
					//float *secondary_hash_projection_data_temp_1 = (float )

					clock_t start = clock();
					//������ת��Ϊ������
					RTOC sRTOC;
					sRTOC.cToR(descriptionsMat_data_temp, descriptionsMat.rows(), descriptionsMat.cols(), descriptionsMat_data_temp_1);
					sRTOC.cToR(primary_hash_projection_data_temp, primary_hash_projection_.rows(), primary_hash_projection_.cols(), primary_hash_projection_data_temp_1);
					//RTOC::cToR(descriptionsMat_data_temp, descriptionsMat.rows(), descriptionsMat.cols(), descriptionsMat_data_temp_1);
					//RTOC::cToR(primary_hash_projection_data_temp, primary_hash_projection_.rows(), primary_hash_projection_.cols(), primary_hash_projection_data_temp_1);

					const float *descriptionsMat_data = const_cast<const float*> (descriptionsMat_data_temp_1);
					const float *primary_hash_projection_data = const_cast<const float*>(primary_hash_projection_data_temp_1);
					//const Eigen::MatrixXf primary_projection = descriptionsMat * primary_hash_projection_;

					int descriptions_size = nbDescriptions * (descriptions.cols());
					int primary_hash_projection_size = (primary_hash_projection_.rows()) * (primary_hash_projection_.cols());
					int result_size = nbDescriptions * (primary_hash_projection_.cols());
					//CPU��Ž����ָ��
					float *mat_result_data = (float*)malloc(result_size * sizeof(float));
					/*
					** GPU ����������
					*/
					dim3 threads(1, 1);
					dim3 grid(1, 1);
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
					//GPU�д�ż������ͼ�������ָ��
					float *descriptionsMat_data_device,
						*primary_hash_projection_data_device,
						*mat_result_device;
					cudaMalloc((void **)&descriptionsMat_data_device, sizeof(float) * descriptions_size);
					cudaMalloc((void **)&primary_hash_projection_data_device, sizeof(float) * primary_hash_projection_size);
					cudaMalloc((void **)&mat_result_device, sizeof(float) * result_size);

					// ���������ݴ��ݽ� �Դ� ���Ѿ����ٺ��˵Ŀռ�
					cudaMemcpy(descriptionsMat_data_device, descriptionsMat_data, sizeof(float) * descriptions_size, cudaMemcpyHostToDevice);
					cudaMemcpy(primary_hash_projection_data_device, primary_hash_projection_data, sizeof(float) * primary_hash_projection_size, cudaMemcpyHostToDevice);

					// ͬ������
					cudaThreadSynchronize();
					// ���ݽ�������˺����еĲ��������庬����ο������ֲᡣ
					float a = 1; float b = 0;
					// ������ˡ��ú�����Ȼ���������������������
					status = cublasSgemm(
						handle,    // blas �����
						CUBLAS_OP_N,    // ���� A ���Բ���
						CUBLAS_OP_N,    // ���� B ���Բ���	
						primary_hash_projection_.cols(),    // B, C ������, n
						descriptions.rows(),    // A, C ������ m
						descriptions.cols(),    // A �������� B ������ k
						&a,    // ����ʽ�� �� ֵ
						primary_hash_projection_data_device,    // B ���Դ��еĵ�ַ
						primary_hash_projection_.cols(),    // B, C ������, n
						descriptionsMat_data_device,    // A ���Դ��еĵ�ַ
						descriptions.cols(),    // ldb k
						&b,    // ����ʽ�� �� ֵ
						mat_result_device,    // C ���Դ��еĵ�ַ(�������)
						primary_hash_projection_.cols()    // B, C ������, n
					);
					cublasDestroy(handle);
					// ����GPU�˼���õĽ��������CPU��
					//int result_size = nbDescriptions * (primary_hash_projection_.cols());
					status = cublasGetMatrix(
						nbDescriptions,//m
						primary_hash_projection_.cols(),//n
						sizeof(*mat_result_data),
						mat_result_device,//d_c
						nbDescriptions,//m
						mat_result_data,//c
						nbDescriptions//m
					);


					for (int i = 0; i < descriptions.rows(); ++i) {
						// Allocate space for each bucket id.
						hashed_descriptions.hashed_desc[i].bucket_ids.resize(nb_bucket_groups_);
						// Compute hash code.
						auto& hash_code = hashed_descriptions.hashed_desc[i].hash_code;
						hash_code = stl::dynamic_bitset(descriptions.cols());
						for (int j = 0; j < nb_hash_code_; ++j)
						{
							hash_code[j] = mat_result_data[(i*nb_hash_code_ + j)] > 0;
						}
					}
					//��ʱ
					clock_t end = clock();
					double endtime = (double)(end - start) / CLOCKS_PER_SEC;

					std::cout << "Total time:" << endtime << std::endl;		//sΪ��λ
					std::cout << "Total time:" << endtime * 1000 << "ms" << std::endl;	//msΪ��λ

																						// �����ʹ�ù����ڴ�

					free(mat_result_data);
					cudaFree(descriptionsMat_data_device);
					cudaFree(primary_hash_projection_data_device);
					cudaFree(mat_result_device);

					// �ͷ� CUBLAS �����
					cublasDestroy(handle);

					// Determine the bucket index for each group.
					Eigen::MatrixXf secondary_projection;
					for (int j = 0; j < nb_bucket_groups_; ++j)
					{
						uint16_t bucket_id = 0;
						secondary_projection = secondary_hash_projection_[j] * descriptionsMat;

						//�д�����
						for (int h = 0; h < descriptionsMat.rows(); h++) {
							for (int k = 0; k < nb_bits_per_bucket_; ++k)
							{
								bucket_id = (bucket_id << 1) + (secondary_projection(h, k) > 0 ? 1 : 0);
							}

							hashed_descriptions.hashed_desc[h].bucket_ids[j] = bucket_id;
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
				//return hashed_descriptions;
			}
			
			void hash_gen
			(
				//��ʽB������
				int mat_I_cols_count,
				//��ʽA���������� ��ʽB��������ΪdescriptionDimension
				int descriptionDimension,
				//��ʽA
				float *primary_hash_projection_data_device,
				//��ʽB
				const float *descriptions_GPU,
				//���C
				float *hash_base_GPU
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
					hash_base_GPU, //�������C
					descriptionDimension//�������C��leading dimension,����Ϊ����������
				);
				cublasDestroy(handle);
			}

			void determine_buket_index_for_each_group
			(
				//������C
				float *secondary_projection_GPU,
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

				status = cublasSgemm
				(
					//A(primary_hash_proejection)
					//B(description_Mat)
					//C(result)
					handle, //blas�����
					CUBLAS_OP_N, //����A��ת��
					CUBLAS_OP_N, //����Bת��
					secondary_rows, //����A��C������,Ҳ�����������
					descrptions_cols, //����B��C��������Ҳ�����������
					descriptionDimension, //����A����������B������
					&a,  //alpha��ֵ
					secondary_hash_projection_j, //�����A
					secondary_rows, //A��leading dimension,����Ϊ����������
					descriptions_GPU, //�Ҿ��� B
					descriptionDimension, //����B��leading dimension,����Ϊ����������
					&b,             //beta��ֵ
					secondary_projection_GPU, //�������C
					secondary_rows//�������C��leading dimension
				);
				cublasDestroy(handle);
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

		
			// Primary hashing function.
			Eigen::MatrixXf primary_hash_projection_;

			// Secondary hashing function.
			std::vector<Eigen::MatrixXf> secondary_hash_projection_;
		};

	}  // namespace matchingGPU
}  // namespace openMVG

#endif // OPENMVG_MATCHING_CASCADE_HASHER_GPU_HPP
   
   
  