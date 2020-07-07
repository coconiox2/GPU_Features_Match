// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2015 Pierre MOULON.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//#include "openMVG/matching_image_collection/Cascade_Hashing_Matcher_Regions.hpp"

#include "Cascade_Hashing_Matcher_Regions_GPU.hpp"
#include "computeMatchesCU.h"
#include "cascade_hasher_GPU.hpp"

//#include "openMVG/matching/cascade_hasher.hpp"
#include "openMVG/features/feature.hpp"
#include "openMVG/matching/matching_filters.hpp"
#include "openMVG/matching/indMatchDecoratorXY.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider.hpp"
#include "openMVG/types.hpp"

#include "third_party/progress/progress.hpp"

namespace openMVG {
	namespace matching_image_collection {

		using namespace openMVG::matching;
		using namespace openMVG::features;
		using namespace computeMatches;

		Cascade_Hashing_Matcher_Regions_GPU
			::Cascade_Hashing_Matcher_Regions_GPU
			(
				float distRatio
			) :Matcher(), f_dist_ratio_(distRatio)
		{
		}

		namespace CascadeHashWithGPU
		{
			template <typename ScalarT>

			///// Portable type used to store an index
			//using IndexT = uint32_t;
			//
			///// Portable value used to save an undefined index value
			//static const IndexT UndefinedIndexT = std::numeric_limits<IndexT>::max();
			//
			///// Standard Pair of IndexT
			//using Pair = std::pair<IndexT, IndexT>;
			//
			///// Set of Pair
			//using Pair_Set = std::set<Pair>;
			//
			///// Vector of Pair
			//using Pair_Vec = std::vector<Pair>;



			void Match
			(
				//offer openMVG::features & openMVG::descriptor
				const sfm::Regions_Provider & regions_provider,
				//pairs of views which need to be match and calculate
				const Pair_Set & pairs,
				//lowe's fliter ratio
				float fDistRatio,
				//pairs of keyPoints computed by cascade
				PairWiseMatchesContainer & map_PutativesMatches, // the pairwise photometric corresponding points
				C_Progress * my_progress_bar
			)
			{
				if (!my_progress_bar)
					my_progress_bar = &C_Progress::dummy();
				my_progress_bar->restart(pairs.size(), "\n- Matching -\n");

				// Collect used view indexes
				std::set<IndexT> used_index;
				// Sort pairs according the first index to minimize later memory swapping
				//std::map use red&black tree to sort it's members automatically
				//IndexT <--> vector,There are multiple views for each view to match
				using Map_vectorT = std::map<IndexT, std::vector<IndexT>>;
				Map_vectorT map_Pairs;
				for (const auto & pair_idx : pairs)
				{
					map_Pairs[pair_idx.first].push_back(pair_idx.second);
					used_index.insert(pair_idx.first);
					used_index.insert(pair_idx.second);
				}

				//A matrix with an element type a, unknown number of rows and columns, and stored as rows
				using BaseMat = Eigen::Matrix<ScalarT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

				// Init the cascade hasher
				CascadeHasher cascade_hasher;
				if (!used_index.empty())
				{
					const IndexT I = *used_index.begin();
					const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
					const size_t dimension = regionsI->DescriptorLength();
					cascade_hasher.Init(dimension);
				}

				std::map<IndexT, HashedDescriptions> hashed_base_;

				// Compute the zero mean descriptor that will be used for hashing (one for all the image regions)
				// A vector of undetermined size but with a value of float data
				Eigen::VectorXf zero_mean_descriptor;
				{
					// A matrix of float type whose size is undetermined
					Eigen::MatrixXf matForZeroMean;
					for (int i = 0; i < used_index.size(); ++i)
					{
						std::set<IndexT>::const_iterator iter = used_index.begin();
						std::advance(iter, i);
						const IndexT I = *iter;
						const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
						//raw data: it seems like return the first descriptor（regionsI's descriptors）'s pointer
						//Regardless of the storage type of the descriptor, it is converted to ScalarT
						const ScalarT * tabI =
							reinterpret_cast<const ScalarT*>(regionsI->DescriptorRawData());
						const size_t dimension = regionsI->DescriptorLength();
						if (i == 0)
						{
							//Each row of the matrix is the size of a descriptor
							matForZeroMean.resize(used_index.size(), dimension);
							matForZeroMean.fill(0.0f);
						}
						if (regionsI->RegionCount() > 0)
						{
							Eigen::Map<BaseMat> mat_I((ScalarT*)tabI, regionsI->RegionCount(), dimension);
							//GPU parallel here may be slower
							//return descriptions.template cast<float>().colwise().mean();
							matForZeroMean.row(i) = CascadeHasher::GetZeroMeanDescriptor(mat_I);
						}
					}
					//GPU parallel here may be slower
					zero_mean_descriptor = CascadeHasher::GetZeroMeanDescriptor(matForZeroMean);
				}

				// Index the input regions
				std::vector<Eigen::Map<BaseMat>> vec_Mat_I;
#ifdef OPENMVG_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif

				for (int i = 0; i < used_index.size(); ++i)
				{
					std::set<IndexT>::const_iterator iter = used_index.begin();
					std::advance(iter, i);
					const IndexT I = *iter;
					const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
					const ScalarT * tabI =
						reinterpret_cast<const ScalarT*>(regionsI->DescriptorRawData());
					const size_t dimension = regionsI->DescriptorLength();

					Eigen::Map<BaseMat> mat_I((ScalarT*)tabI, regionsI->RegionCount(), dimension);
					vec_Mat_I.push_back(mat_I);
//#ifdef OPENMVG_USE_OPENMP
//#pragma omp critical
//#endif
//					{
//						cascade_hasher.CreateHashedDescriptions(mat_I, zero_mean_descriptor);
//					}
				}
				cascade_hasher.CreateHashedDescriptions(vec_Mat_I, zero_mean_descriptor, hashed_base_);

				/*for (int i = 0; i < used_index.size(); ++i)
				{
					std::set<IndexT>::const_iterator iter = used_index.begin();
					std::advance(iter, i);
					const IndexT I = *iter;
					const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
					const ScalarT * tabI =
						reinterpret_cast<const ScalarT*>(regionsI->DescriptorRawData());
					const size_t dimension = regionsI->DescriptorLength();

					Eigen::Map<BaseMat> mat_I((ScalarT*)tabI, regionsI->RegionCount(), dimension);
#ifdef OPENMVG_USE_OPENMP
#pragma omp critical
#endif
					{
						hashed_base_[I] =
							std::move(cascade_hasher.CreateHashedDescriptions(mat_I, zero_mean_descriptor));
					}
				}*/

				// Perform matching between all the pairs
				for (const auto & pairs : map_Pairs)
				{
					if (my_progress_bar->hasBeenCanceled())
						break;
					const IndexT I = pairs.first;
					const std::vector<IndexT> & indexToCompare = pairs.second;

					const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
					if (regionsI->RegionCount() == 0)
					{
						(*my_progress_bar) += indexToCompare.size();
						continue;
					}

					const std::vector<features::PointFeature> pointFeaturesI = regionsI->GetRegionsPositions();
					const ScalarT * tabI =
						reinterpret_cast<const ScalarT*>(regionsI->DescriptorRawData());
					const size_t dimension = regionsI->DescriptorLength();
					Eigen::Map<BaseMat> mat_I((ScalarT*)tabI, regionsI->RegionCount(), dimension);

#ifdef OPENMVG_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
					for (int j = 0; j < (int)indexToCompare.size(); ++j)
					{
						if (my_progress_bar->hasBeenCanceled())
							continue;
						const size_t J = indexToCompare[j];
						const std::shared_ptr<features::Regions> regionsJ = regions_provider.get(J);

						if (regionsI->Type_id() != regionsJ->Type_id())
						{
							++(*my_progress_bar);
							continue;
						}

						// Matrix representation of the query input data;
						const ScalarT * tabJ = reinterpret_cast<const ScalarT*>(regionsJ->DescriptorRawData());
						Eigen::Map<BaseMat> mat_J((ScalarT*)tabJ, regionsJ->RegionCount(), dimension);

						IndMatches pvec_indices;
						using ResultType = typename Accumulator<ScalarT>::Type;
						std::vector<ResultType> pvec_distances;
						pvec_distances.reserve(regionsJ->RegionCount() * 2);
						pvec_indices.reserve(regionsJ->RegionCount() * 2);

						// Match the query descriptors to the database
						cascade_hasher.Match_HashedDescriptions<BaseMat, ResultType>(
							hashed_base_[J], mat_J,
							hashed_base_[I], mat_I,
							&pvec_indices, &pvec_distances);

						std::vector<int> vec_nn_ratio_idx;
						//从这里把GPU计算得到的匹配再传回到CPU上来做滤波
						// Filter the matches using a distance ratio test:
						//   The probability that a match is correct is determined by taking
						//   the ratio of distance from the closest neighbor to the distance
						//   of the second closest.
						matching::NNdistanceRatio(
							pvec_distances.begin(), // distance start
							pvec_distances.end(),   // distance end
							2, // Number of neighbor in iterator sequence (minimum required 2)
							vec_nn_ratio_idx, // output (indices that respect the distance Ratio)
							Square(fDistRatio));

						matching::IndMatches vec_putative_matches;
						vec_putative_matches.reserve(vec_nn_ratio_idx.size());
						for (size_t k = 0; k < vec_nn_ratio_idx.size(); ++k)
						{
							const size_t index = vec_nn_ratio_idx[k];
							vec_putative_matches.emplace_back(pvec_indices[index * 2].j_, pvec_indices[index * 2].i_);
						}

						// Remove duplicates
						matching::IndMatch::getDeduplicated(vec_putative_matches);

						// Remove matches that have the same (X,Y) coordinates
						const std::vector<features::PointFeature> pointFeaturesJ = regionsJ->GetRegionsPositions();
						matching::IndMatchDecorator<float> matchDeduplicator(vec_putative_matches,
							pointFeaturesI, pointFeaturesJ);
						matchDeduplicator.getDeduplicated(vec_putative_matches);

#ifdef OPENMVG_USE_OPENMP
#pragma omp critical
#endif
						{
							if (!vec_putative_matches.empty())
							{
								map_PutativesMatches.insert(
								{
									{ I,J },
									std::move(vec_putative_matches)
								});
							}
						}
						++(*my_progress_bar);
					}
				}
			}
		} // namespace CascadeHashWithGPU

		

		void Cascade_Hashing_Matcher_Regions_GPU::Match
		(
			const std::shared_ptr<sfm::Regions_Provider> & regions_provider,
			const Pair_Set & pairs,
			PairWiseMatchesContainer & map_PutativesMatches, // the pairwise photometric corresponding points
			C_Progress * my_progress_bar
		)const
		{
#ifdef OPENMVG_USE_OPENMP
			std::cout << "Using the OPENMP thread interface" << std::endl;
#endif
			if (!regions_provider)
				return;

			if (regions_provider->IsBinary())
				return;
			
			
			//cascade with GPU accelerate
			if (computeMatches::computeMatchesWithGPU)
			{
				if (regions_provider->Type_id() == typeid(unsigned char).name())
				{
					CascadeHashWithGPU::Match<unsigned char>(
						*regions_provider.get(),
						pairs,
						f_dist_ratio_,
						map_PutativesMatches,
						my_progress_bar);
				}
				else
					if (regions_provider->Type_id() == typeid(float).name())
					{
						CascadeHashWithGPU::Match<float>(
							*regions_provider.get(),
							pairs,
							f_dist_ratio_,
							map_PutativesMatches,
							my_progress_bar);
					}
					else
					{
						std::cerr << "Matcher not implemented for this region type" << std::endl;
					}
			}
			else
			{
				std::cerr << "error when setting bool computeMatches::computeMatchesWithGPU! \n" << std::endl;
				return;
			}
		}

	} // namespace openMVG
} // namespace matching_image_collection
