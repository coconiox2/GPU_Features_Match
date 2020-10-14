#include "utils.hpp"
#include "computeMatchesCU.h"
//#include "Cascade_Hashing_Matcher_Regions_GPU.hpp"
#include "cascade_hasher_GPU.hpp"

//openMVG
#include <Eigen/Core>

#include "openMVG/graph/graph.hpp"
#include "openMVG/graph/graph_stats.hpp"

#include "openMVG/image/image_io.hpp"
#include "openMVG/image/image_concat.hpp"
#include "openMVG/matching/svg_matches.hpp"

#include "openMVG/graph/graph.hpp"
#include "openMVG/features/akaze/image_describer_akaze.hpp"
#include "openMVG/features/descriptor.hpp"
#include "openMVG/features/feature.hpp"
#include "openMVG/matching/indMatch.hpp"
#include "openMVG/matching/indMatch_utils.hpp"

#include "openMVG/matching/indMatchDecoratorXY.hpp"
#include "openMVG/matching/matching_filters.hpp"

#include "openMVG/matching_image_collection/Matcher_Regions.hpp"
#include "openMVG/matching_image_collection/Cascade_Hashing_Matcher_Regions.hpp"
#include "openMVG/matching_image_collection/GeometricFilter.hpp"
#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider.hpp"
#include "openMVG/types.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider_cache.hpp"
#include "openMVG/matching_image_collection/F_ACRobust.hpp"
#include "openMVG/matching_image_collection/E_ACRobust.hpp"
#include "openMVG/matching_image_collection/E_ACRobust_Angular.hpp"
#include "openMVG/matching_image_collection/Eo_Robust.hpp"
#include "openMVG/matching_image_collection/H_ACRobust.hpp"
#include "openMVG/matching_image_collection/Pair_Builder.hpp"
#include "openMVG/matching/pairwiseAdjacencyDisplay.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"
#include "openMVG/stl/stl.hpp"
#include "openMVG/system/timer.hpp"

#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"

//CUDA V 10.2
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::matching;
using namespace openMVG::robust;
using namespace openMVG::sfm;
using namespace openMVG::matching_image_collection;
using namespace std;

using BaseMat = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;


extern "C" int testCUDACPP(int a, int b);

void computeMatches::test() {
	int a = 1;
	int b = 2;
	int c = -1;
	c = testCUDACPP(a, b);
	printf("testCUDACPP result:%d", c);
	printf("cpp test success\n");
}

void computeMatches::computeZeroMeanDescriptors
(
	Eigen::VectorXf &_zero_mean_descriptor,		//输出结果
	const sfm::Regions_Provider & regions_provider,
	const Pair_Set & pairs
) 
{
	std::cout << "Compute zero_mean_descriptor begin: " << std::endl;
	system::Timer timeComputeZeroMeanDescriptor;

	// Collect used view indexes
	std::set<IndexT> used_index;
	// Sort pairs according the first index to minimize later memory swapping
	using Map_vectorT = std::map<IndexT, std::vector<IndexT>>;
	Map_vectorT map_Pairs;
	for (const auto & pair_idx : pairs)
	{
		map_Pairs[pair_idx.first].push_back(pair_idx.second);
		used_index.insert(pair_idx.first);
		used_index.insert(pair_idx.second);
	}

	//初始化cascade hasher
	CascadeHasherGPU zeroCascadeHasher;
	zeroCascadeHasher.Init(descriptionDimension);

	// Compute the zero mean descriptor that will be used for hashing (one for all the image regions)
	Eigen::VectorXf zero_mean_descriptor;
	{
		Eigen::MatrixXf matForZeroMean;
		for (int i = 0; i < used_index.size(); ++i)
		{
			std::set<IndexT>::const_iterator iter = used_index.begin();
			std::advance(iter, i);
			const IndexT I = *iter;
			const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
			const unsigned char * tabI =
				reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
			const size_t dimension = regionsI->DescriptorLength();
			if (i == 0)
			{
				matForZeroMean.resize(dimension, used_index.size());
				matForZeroMean.fill(0.0f);
			}
			if (regionsI->RegionCount() > 0)
			{
				Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
				matForZeroMean.col(i) = zeroCascadeHasher.GetZeroMeanDescriptor(mat_I);
			}
		}
		zero_mean_descriptor = zeroCascadeHasher.GetZeroMeanDescriptor(matForZeroMean);
	}
	std::cout << "Task (Compute zero_mean_descriptor) done in (s): " << timeComputeZeroMeanDescriptor.elapsed() << std::endl;
}

void computeMatches::computeHashes
(
	std::vector<Eigen::Map<BaseMat>> mat_I_vec,
	std::map<IndexT, HashedDescriptions> hashed_base_,
	const Eigen::VectorXf & zero_mean_descriptor,
	// The number of bucket bits.
	int nb_bits_per_bucket_,
	// The number of dimensions of the Hash code.
	int nb_hash_code_,
	// The number of bucket groups.
	int nb_bucket_groups_,
	// The number of buckets in each group.
	int nb_buckets_per_group_,
	Eigen::MatrixXf primary_hash_projection_,
	std::vector<Eigen::MatrixXf> secondary_hash_projection_
) 
{
	CascadeHasherGPU cascadeHasher;
	cascadeHasher.Init(descriptionDimension);


	// Create hash codes for each description.
	for(int i = 0; i < mat_I_vec.size(); i++)
	{
		// Allocate space for hash codes.
		const typename Eigen::Map<BaseMat>::Index nbDescriptions = mat_I_vec[i].rows();
		hashed_base_[i].hashed_desc.resize(nbDescriptions);
		Eigen::VectorXf descriptor(mat_I_vec[i].cols());
		for (int i = 0; i < nbDescriptions; ++i)
		{
			// Allocate space for each bucket id.
			hashed_base_[i].hashed_desc[i].bucket_ids.resize(nb_bucket_groups_);

			// Compute hash code.
			auto& hash_code = hashed_base_[i].hashed_desc[i].hash_code;
			hash_code = stl::dynamic_bitset(mat_I_vec[i].cols());
			descriptor = mat_I_vec[i].row(i).template cast<float>();
			descriptor -= zero_mean_descriptor;
			const Eigen::VectorXf primary_projection = primary_hash_projection_ * descriptor;
			for (int j = 0; j < nb_hash_code_; ++j)
			{
				hash_code[j] = primary_projection(j) > 0;
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
				hashed_base_[i].hashed_desc[i].bucket_ids[j] = bucket_id;
			}
		}
	}
}

int computeMatches::computeMatchesGPU
(
	const sfm::Regions_Provider & regions_provider,
	Pair_Set &pairs,
	PairWiseMatches &map_PutativesMatches,
	C_Progress *my_progress_bar
)
{
	if (!my_progress_bar)
		my_progress_bar = &C_Progress::dummy();
	my_progress_bar->restart(pairs.size(), "\n- Matching -\n");

	// Collect used view indexes
	std::set<IndexT> used_index;
	// Sort pairs according the first index to minimize later memory swapping
	using Map_vectorT = std::map<IndexT, std::vector<IndexT>>;
	Map_vectorT map_Pairs;
	for (const auto & pair_idx : pairs)
	{
		map_Pairs[pair_idx.first].push_back(pair_idx.second);
		used_index.insert(pair_idx.first);
		used_index.insert(pair_idx.second);
	}

	// Init the cascade hasher
	CascadeHasherGPU cascade_hasher;
	cascade_hasher.Init(descriptionDimension);

	std::map<IndexT, HashedDescriptions> hashed_base_;
	
	//计算零和平均描述符
	Eigen::VectorXf zero_mean_descriptor;
	computeMatches::computeZeroMeanDescriptors(zero_mean_descriptor, regions_provider, pairs);

	//从内存中一次性读出来所有的哈希描述符数据，用于并行调用GPU进行计算
	std::vector<Eigen::Map<BaseMat>> mat_I_vec;
	// Index the input regions
#ifdef OPENMVG_USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif

	for (int i = 0; i < used_index.size(); ++i)
	{
		std::set<IndexT>::const_iterator iter = used_index.begin();
		std::advance(iter, i);
		const IndexT I = *iter;
		const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
		const unsigned char * tabI =
			reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
		const size_t dimension = regionsI->DescriptorLength();

		Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());

		mat_I_vec[i] = std::move(mat_I);
	}

	computeMatches::computeHashes
	(
		mat_I_vec, 
		hashed_base_, 
		zero_mean_descriptor, 
		cascade_hasher.nb_bits_per_bucket_,
		cascade_hasher.nb_hash_code_,
		cascade_hasher.nb_bucket_groups_,
		cascade_hasher.nb_buckets_per_group_,
		cascade_hasher.primary_hash_projection_,
		cascade_hasher.secondary_hash_projection_
	);
}

int computeMatches::computeMatchesMVG(std::string sSfM_Data_FilenameDir_father)
{
	std::string sSfM_Data_Filename = sSfM_Data_FilenameDir_father + "/sfm_data.json";
	std::string sMatchesDirectory = sSfM_Data_FilenameDir_father;
	std::string sGeometricModel = "e";
	float fDistRatio = 0.8f;
	int iMatchingVideoMode = -1;
	std::string sPredefinedPairList = sSfM_Data_FilenameDir_father + "/pair_list.txt";
	std::string sNearestMatchingMethod = "AUTO";
	bool bForce = false;
	bool bGuided_matching = false;
	int imax_iteration = 2048;
	unsigned int ui_max_cache_size = 0;

	

	std::cout << " You called : " << "\n"
		<< "--input_file " << sSfM_Data_Filename << "\n"
		<< "--out_dir " << sMatchesDirectory << "\n"
		<< "Optional parameters:" << "\n"
		<< "--force " << bForce << "\n"
		<< "--ratio " << fDistRatio << "\n"
		<< "--geometric_model " << sGeometricModel << "\n"
		<< "--video_mode_matching " << iMatchingVideoMode << "\n"
		<< "--pair_list " << sPredefinedPairList << "\n"
		<< "--nearest_matching_method " << sNearestMatchingMethod << "\n"
		<< "--guided_matching " << bGuided_matching << "\n"
		<< "--cache_size " << ((ui_max_cache_size == 0) ? "unlimited" : std::to_string(ui_max_cache_size)) << std::endl;

	EPairMode ePairmode = (iMatchingVideoMode == -1) ? PAIR_EXHAUSTIVE : PAIR_CONTIGUOUS;

	if (sPredefinedPairList.length()) {
		ePairmode = PAIR_FROM_FILE;
		if (iMatchingVideoMode>0) {
			std::cerr << "\nIncompatible options: --videoModeMatching and --pairList" << std::endl;
			return EXIT_FAILURE;
		}
	}

	if (sMatchesDirectory.empty() || !stlplus::is_folder(sMatchesDirectory)) {
		std::cerr << "\nIt is an invalid output directory" << std::endl;
		return EXIT_FAILURE;
	}

	EGeometricModel eGeometricModelToCompute = FUNDAMENTAL_MATRIX;
	std::string sGeometricMatchesFilename = "";
	switch (sGeometricModel[0])
	{
	case 'f': case 'F':
		eGeometricModelToCompute = FUNDAMENTAL_MATRIX;
		sGeometricMatchesFilename = "matches.f.bin";
		break;
	case 'e': case 'E':
		eGeometricModelToCompute = ESSENTIAL_MATRIX;
		sGeometricMatchesFilename = "matches.e.bin";
		break;
	case 'h': case 'H':
		eGeometricModelToCompute = HOMOGRAPHY_MATRIX;
		sGeometricMatchesFilename = "matches.h.bin";
		break;
	case 'a': case 'A':
		eGeometricModelToCompute = ESSENTIAL_MATRIX_ANGULAR;
		sGeometricMatchesFilename = "matches.f.bin";
		break;
	case 'o': case 'O':
		eGeometricModelToCompute = ESSENTIAL_MATRIX_ORTHO;
		sGeometricMatchesFilename = "matches.o.bin";
		break;
	case 'u': case 'U':
		eGeometricModelToCompute = ESSENTIAL_MATRIX_UPRIGHT;
		sGeometricMatchesFilename = "matches.f.bin";
		break;
	default:
		std::cerr << "Unknown geometric model" << std::endl;
		return EXIT_FAILURE;
	}

	// -----------------------------
	// - Load SfM_Data Views & intrinsics data
	// a. Compute putative descriptor matches
	// b. Geometric filtering of putative matches
	// + Export some statistics
	// -----------------------------

	//---------------------------------------
	// Read SfM Scene (image view & intrinsics data)
	//---------------------------------------
	SfM_Data sfm_data;
	if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(VIEWS | INTRINSICS))) {
		std::cerr << std::endl
			<< "The input SfM_Data file \"" << sSfM_Data_Filename << "\" cannot be read." << std::endl;
		return EXIT_FAILURE;
	}

	//---------------------------------------
	// Load SfM Scene regions
	//---------------------------------------
	// Init the regions_type from the image describer file (used for image regions extraction)
	using namespace openMVG::features;
	const std::string sImage_describer = stlplus::create_filespec(sMatchesDirectory, "image_describer", "json");
	std::unique_ptr<Regions> regions_type = Init_region_type_from_file(sImage_describer);
	if (!regions_type)
	{
		std::cerr << "Invalid: "
			<< sImage_describer << " regions type file." << std::endl;
		return EXIT_FAILURE;
	}

	//---------------------------------------
	// a. Compute putative descriptor matches
	//    - Descriptor matching (according user method choice)
	//    - Keep correspondences only if NearestNeighbor ratio is ok
	//---------------------------------------

	// Load the corresponding view regions
	std::shared_ptr<Regions_Provider> regions_provider;
	if (ui_max_cache_size == 0)
	{
		// Default regions provider (load & store all regions in memory)
		regions_provider = std::make_shared<Regions_Provider>();
	}
	else
	{
		// Cached regions provider (load & store regions on demand)
		regions_provider = std::make_shared<Regions_Provider_Cache>(ui_max_cache_size);
	}

	// Show the progress on the command line:
	C_Progress_display progress;

	if (!regions_provider->load(sfm_data, sMatchesDirectory, regions_type, &progress)) {
		std::cerr << std::endl << "Invalid regions." << std::endl;
		return EXIT_FAILURE;
	}

	PairWiseMatches map_PutativesMatches;

	// Build some alias from SfM_Data Views data:
	// - List views as a vector of filenames & image sizes
	std::vector<std::string> vec_fileNames;
	std::vector<std::pair<size_t, size_t>> vec_imagesSize;
	{
		vec_fileNames.reserve(sfm_data.GetViews().size());
		vec_imagesSize.reserve(sfm_data.GetViews().size());
		for (Views::const_iterator iter = sfm_data.GetViews().begin();
			iter != sfm_data.GetViews().end();
			++iter)
		{
			const View * v = iter->second.get();
			vec_fileNames.push_back(stlplus::create_filespec(sfm_data.s_root_path,
				v->s_Img_path));
			vec_imagesSize.push_back(std::make_pair(v->ui_width, v->ui_height));
		}
	}

	std::cout << std::endl << " - PUTATIVE MATCHES - " << std::endl;
	// If the matches already exists, reload them
	if (!bForce
		&& (stlplus::file_exists(sMatchesDirectory + "/matches.putative.txt")
			|| stlplus::file_exists(sMatchesDirectory + "/matches.putative.bin"))
		)
	{
		if (!(Load(map_PutativesMatches, sMatchesDirectory + "/matches.putative.bin") ||
			Load(map_PutativesMatches, sMatchesDirectory + "/matches.putative.txt")))
		{
			std::cerr << "Cannot load input matches file";
			return EXIT_FAILURE;
		}
		std::cout << "\t PREVIOUS RESULTS LOADED;"
			<< " #pair: " << map_PutativesMatches.size() << std::endl;
	}
	else // Compute the putative matches
	{
		std::cout << "Use: ";
		switch (ePairmode)
		{
		case PAIR_EXHAUSTIVE: std::cout << "exhaustive pairwise matching" << std::endl; break;
		case PAIR_CONTIGUOUS: std::cout << "sequence pairwise matching" << std::endl; break;
		case PAIR_FROM_FILE:  std::cout << "user defined pairwise matching" << std::endl; break;
		}

		// Allocate the right Matcher according the Matching requested method
		std::unique_ptr<Matcher> collectionMatcher;
		if (sNearestMatchingMethod == "AUTO")
		{
			if (regions_type->IsScalar())
			{
				std::cout << "Using FAST_CASCADE_HASHING_L2 matcher" << std::endl;
				collectionMatcher.reset(new Cascade_Hashing_Matcher_Regions(fDistRatio));
			}
			else
				if (regions_type->IsBinary())
				{
					std::cout << "Using BRUTE_FORCE_HAMMING matcher" << std::endl;
					collectionMatcher.reset(new Matcher_Regions(fDistRatio, BRUTE_FORCE_HAMMING));
				}
		}
		else
			if (sNearestMatchingMethod == "BRUTEFORCEL2")
			{
				std::cout << "Using BRUTE_FORCE_L2 matcher" << std::endl;
				collectionMatcher.reset(new Matcher_Regions(fDistRatio, BRUTE_FORCE_L2));
			}
			else
				if (sNearestMatchingMethod == "BRUTEFORCEHAMMING")
				{
					std::cout << "Using BRUTE_FORCE_HAMMING matcher" << std::endl;
					collectionMatcher.reset(new Matcher_Regions(fDistRatio, BRUTE_FORCE_HAMMING));
				}
				else
					if (sNearestMatchingMethod == "HNSWL2")
					{
						std::cout << "Using HNSWL2 matcher" << std::endl;
						collectionMatcher.reset(new Matcher_Regions(fDistRatio, HNSW_L2));
					}
					else
						if (sNearestMatchingMethod == "ANNL2")
						{
							std::cout << "Using ANN_L2 matcher" << std::endl;
							collectionMatcher.reset(new Matcher_Regions(fDistRatio, ANN_L2));
						}
						else
							if (sNearestMatchingMethod == "CASCADEHASHINGL2")
							{
								std::cout << "Using CASCADE_HASHING_L2 matcher" << std::endl;
								collectionMatcher.reset(new Matcher_Regions(fDistRatio, CASCADE_HASHING_L2));
							}
							else
								if (sNearestMatchingMethod == "FASTCASCADEHASHINGL2")
								{
									std::cout << "Using FAST_CASCADE_HASHING_L2 matcher" << std::endl;
									collectionMatcher.reset(new Cascade_Hashing_Matcher_Regions(fDistRatio));
								}
		if (!collectionMatcher)
		{
			std::cerr << "Invalid Nearest Neighbor method: " << sNearestMatchingMethod << std::endl;
			return EXIT_FAILURE;
		}
		// Perform the matching
		system::Timer timer;
		{
			// From matching mode compute the pair list that have to be matched:
			Pair_Set pairs;
			switch (ePairmode)
			{
			case PAIR_EXHAUSTIVE: pairs = exhaustivePairs(sfm_data.GetViews().size()); break;
			case PAIR_CONTIGUOUS: pairs = contiguousWithOverlap(sfm_data.GetViews().size(), iMatchingVideoMode); break;
			case PAIR_FROM_FILE:
				if (!loadPairs(sfm_data.GetViews().size(), sPredefinedPairList, pairs))
				{
					return EXIT_FAILURE;
				}
				break;
			}
			// Photometric matching of putative pairs
			computeMatches::computeMatchesGPU(*regions_provider.get(), pairs, map_PutativesMatches, &progress);
			//---------------------------------------
			//-- Export putative matches
			//---------------------------------------
			if (!Save(map_PutativesMatches, std::string(sMatchesDirectory + "/matches.putative.bin")))
			{
				std::cerr
					<< "Cannot save computed matches in: "
					<< std::string(sMatchesDirectory + "/matches.putative.bin");
				return EXIT_FAILURE;
			}
		}
		std::cout << "Task (Regions Matching) done in (s): " << timer.elapsed() << std::endl;
	}
	//-- export putative matches Adjacency matrix
	PairWiseMatchingToAdjacencyMatrixSVG(vec_fileNames.size(),
		map_PutativesMatches,
		stlplus::create_filespec(sMatchesDirectory, "PutativeAdjacencyMatrix", "svg"));
	//-- export view pair graph once putative graph matches have been computed
	{
		std::set<IndexT> set_ViewIds;
		std::transform(sfm_data.GetViews().begin(), sfm_data.GetViews().end(),
			std::inserter(set_ViewIds, set_ViewIds.begin()), stl::RetrieveKey());
		graph::indexedGraph putativeGraph(set_ViewIds, getPairs(map_PutativesMatches));
		graph::exportToGraphvizData(
			stlplus::create_filespec(sMatchesDirectory, "putative_matches"),
			putativeGraph);
	}

	//---------------------------------------
	// b. Geometric filtering of putative matches
	//    - AContrario Estimation of the desired geometric model
	//    - Use an upper bound for the a contrario estimated threshold
	//---------------------------------------

	std::unique_ptr<ImageCollectionGeometricFilter> filter_ptr(
		new ImageCollectionGeometricFilter(&sfm_data, regions_provider));

	if (filter_ptr)
	{
		system::Timer timer;
		const double d_distance_ratio = 0.6;

		PairWiseMatches map_GeometricMatches;
		switch (eGeometricModelToCompute)
		{
		case HOMOGRAPHY_MATRIX:
		{
			const bool bGeometric_only_guided_matching = true;
			filter_ptr->Robust_model_estimation(
				GeometricFilter_HMatrix_AC(4.0, imax_iteration),
				map_PutativesMatches, bGuided_matching,
				bGeometric_only_guided_matching ? -1.0 : d_distance_ratio, &progress);
			map_GeometricMatches = filter_ptr->Get_geometric_matches();
		}
		break;
		case FUNDAMENTAL_MATRIX:
		{
			filter_ptr->Robust_model_estimation(
				GeometricFilter_FMatrix_AC(4.0, imax_iteration),
				map_PutativesMatches, bGuided_matching, d_distance_ratio, &progress);
			map_GeometricMatches = filter_ptr->Get_geometric_matches();
		}
		break;
		case ESSENTIAL_MATRIX:
		{
			filter_ptr->Robust_model_estimation(
				GeometricFilter_EMatrix_AC(4.0, imax_iteration),
				map_PutativesMatches, bGuided_matching, d_distance_ratio, &progress);
			map_GeometricMatches = filter_ptr->Get_geometric_matches();

			//-- Perform an additional check to remove pairs with poor overlap
			std::vector<PairWiseMatches::key_type> vec_toRemove;
			for (const auto & pairwisematches_it : map_GeometricMatches)
			{
				const size_t putativePhotometricCount = map_PutativesMatches.find(pairwisematches_it.first)->second.size();
				const size_t putativeGeometricCount = pairwisematches_it.second.size();
				const float ratio = putativeGeometricCount / static_cast<float>(putativePhotometricCount);
				if (putativeGeometricCount < 50 || ratio < .3f) {
					// the pair will be removed
					vec_toRemove.push_back(pairwisematches_it.first);
				}
			}
			//-- remove discarded pairs
			for (const auto & pair_to_remove_it : vec_toRemove)
			{
				map_GeometricMatches.erase(pair_to_remove_it);
			}
		}
		break;
		case ESSENTIAL_MATRIX_ANGULAR:
		{
			filter_ptr->Robust_model_estimation(
				GeometricFilter_ESphericalMatrix_AC_Angular<false>(4.0, imax_iteration),
				map_PutativesMatches, bGuided_matching, d_distance_ratio, &progress);
			map_GeometricMatches = filter_ptr->Get_geometric_matches();
		}
		break;
		case ESSENTIAL_MATRIX_ORTHO:
		{
			filter_ptr->Robust_model_estimation(
				GeometricFilter_EOMatrix_RA(2.0, imax_iteration),
				map_PutativesMatches, bGuided_matching, d_distance_ratio, &progress);
			map_GeometricMatches = filter_ptr->Get_geometric_matches();
		}
		break;
		case ESSENTIAL_MATRIX_UPRIGHT:
		{
			filter_ptr->Robust_model_estimation(
				GeometricFilter_ESphericalMatrix_AC_Angular<true>(4.0, imax_iteration),
				map_PutativesMatches, bGuided_matching, d_distance_ratio, &progress);
			map_GeometricMatches = filter_ptr->Get_geometric_matches();
		}
		break;
		}

		//---------------------------------------
		//-- Export geometric filtered matches
		//---------------------------------------
		if (!Save(map_GeometricMatches,
			std::string(sMatchesDirectory + "/" + sGeometricMatchesFilename)))
		{
			std::cerr
				<< "Cannot save computed matches in: "
				<< std::string(sMatchesDirectory + "/" + sGeometricMatchesFilename);
			return EXIT_FAILURE;
		}

		std::cout << "Task done in (s): " << timer.elapsed() << std::endl;

		// -- export Geometric View Graph statistics
		graph::getGraphStatistics(sfm_data.GetViews().size(), getPairs(map_GeometricMatches));

		//-- export Adjacency matrix
		std::cout << "\n Export Adjacency Matrix of the pairwise's geometric matches"
			<< std::endl;
		PairWiseMatchingToAdjacencyMatrixSVG(vec_fileNames.size(),
			map_GeometricMatches,
			stlplus::create_filespec(sMatchesDirectory, "GeometricAdjacencyMatrix", "svg"));

		//-- export view pair graph once geometric filter have been done
		{
			std::set<IndexT> set_ViewIds;
			std::transform(sfm_data.GetViews().begin(), sfm_data.GetViews().end(),
				std::inserter(set_ViewIds, set_ViewIds.begin()), stl::RetrieveKey());
			graph::indexedGraph putativeGraph(set_ViewIds, getPairs(map_GeometricMatches));
			graph::exportToGraphvizData(
				stlplus::create_filespec(sMatchesDirectory, "geometric_matches"),
				putativeGraph);
		}
	}
	return EXIT_SUCCESS;
}
