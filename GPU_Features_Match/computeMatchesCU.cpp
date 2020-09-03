#include "utils.hpp"
#include "computeMatchesCU.h"
//#include "Cascade_Hashing_Matcher_Regions_GPU.hpp"
#include "cascade_hasher_GPU.hpp"

//openMVG

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
#include "third_party/cmdLine/cmdLine.h"
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

//#include "third_party/eigen/eigen/src/Core/util/Macros.h"
void computeMatches::test() {
	int a = 1;
	int b = 2;
	int c = -1;
	c = testCUDACPP(a, b);
	printf("testCUDACPP result:%d", c);
	printf("cpp test success\n");
}
void computeMatches::computeZeroMeanDescriptors(Eigen::VectorXf &_zero_mean_descriptor)
{
	int imgCount = group_count*block_count_per_group*image_count_per_block;
	//用于保存每组数据计算得到的零和平均值，最后再平均为zero_mean_descriptors
	Eigen::MatrixXf zero_descriptor;
	zero_descriptor.resize(descriptionDimension, group_count);
	zero_descriptor.fill(0.0f);

	//初始化cascade hasher
	CascadeHasherGPU zeroCascadeHasher;
	zeroCascadeHasher.Init(descriptionDimension);

	std::string sSfM_Data_Filename;
	std::string sMatchesOutputDir;
	char temp_i[2] = { ' ','\0' };

	std::cout << "Compute zero_mean_descriptor begin: " << std::endl;
	system::Timer timer;
	for (int i = 0; i < group_count; i++) {
		temp_i[0] = i + 48;
		const std::string str_i = temp_i;
		sSfM_Data_Filename = sSfM_Data_FilenameDir_father + "DJI_" + str_i + "_build/" + "sfm_data.json";
		sMatchesOutputDir = sMatchesOutputDir_father + "DJI_" + str_i + "_build/";
		//---------------------------------------
		// Read SfM Scene (image view & intrinsics data)
		//---------------------------------------
		SfM_Data sfm_data;
		if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(VIEWS | INTRINSICS))) {
			std::cerr << std::endl
				<< "The input SfM_Data file \"" << sSfM_Data_Filename << "\" cannot be read." << std::endl;
			return;
		}
		//---------------------------------------
		// Load SfM Scene regions
		//---------------------------------------
		// Init the regions_type from the image describer file (used for image regions extraction)
		using namespace openMVG::features;
		const std::string sImage_describer = stlplus::create_filespec(sMatchesOutputDir, "image_describer", "json");
		std::unique_ptr<Regions> regions_type = Init_region_type_from_file(sImage_describer);
		if (!regions_type)
		{
			std::cerr << "Invalid: "
				<< sImage_describer << " regions type file." << std::endl;
			return;
		}
		// Load the corresponding view regions
		std::shared_ptr<Regions_Provider> regions_provider;
		// Default regions provider (load & store all regions in memory)
		regions_provider = std::make_shared<Regions_Provider>();
		// Show the progress on the command line:
		C_Progress_display progress;

		if (!regions_provider->load(sfm_data, sMatchesOutputDir, regions_type, &progress)) {
			std::cerr << std::endl << "Invalid regions." << std::endl;
			return;
		}
		//存储每一组内的均值
		Eigen::MatrixXf matForZeroMean;
		for (int j = 0; j < image_count_per_group; j++) {
			const IndexT I = j;
			const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
			const unsigned char * tabI =
				reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());

			const size_t dimension = regionsI->DescriptorLength();
			if (j == 0)
			{
				//Each row of the matrix is the size of a descriptor
				matForZeroMean.resize(dimension, image_count_per_group);
				matForZeroMean.fill(0.0f);
			}
			if (regionsI->RegionCount() > 0)
			{
				Eigen::Map<BaseMat> mat_I((unsigned char* )tabI, dimension, regionsI->RegionCount());
				matForZeroMean.col(i) = zeroCascadeHasher.GetZeroMeanDescriptor(mat_I);
			}
		}
		zero_descriptor.col(i) = zeroCascadeHasher.GetZeroMeanDescriptor(matForZeroMean);
	}
	_zero_mean_descriptor = zeroCascadeHasher.GetZeroMeanDescriptor(zero_descriptor);
	std::cout << "Task (Compute zero_mean_descriptor) done in (s): " << timer.elapsed() << std::endl;
}
//void computeMatches::computeCurrentGroupHashcode() {}
//void computeMatches::computeCurrentBlockHashcode
//(
//	int secondIter,
//	std::vector <int> mat_I_cols,
//	float **hash_base_array_GPU,
//	CascadeHasher myCascadeHasher,
//	float *primary_hash_projection_data_device,
//	float **mat_I_point_array_GPU,
//	float **hash_base_array_CPU,
//	const float **mat_I_point_array_CPU,
//	std::map<openMVG::IndexT, HashedDescriptions> &hashed_base_,
//	float **secondary_hash_projection_data_GPU,
//	string sMatchesOutputDir_hash
//)
//{
//	//for (int m = 0; m < image_count_per_block; ++m) {
//	//	int hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
//	//	cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
//	//	myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
//	//		primary_hash_projection_data_device, mat_I_point_array_GPU[m],
//	//		hash_base_array_GPU[m]);
//	//	cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);
//
//	//	//free
//	//	{
//	//		cudaFree(hash_base_array_GPU[m]);
//	//		mat_I_point_array_CPU[m] = NULL;
//	//		hash_base_array_GPU[m] = NULL;
//	//	}
//
//
//	//	//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面
//	//	{
//	//		for (int i = 0; i < mat_I_cols[m]; ++i) {
//	//			// Allocate space for each bucket id.
//	//			hashed_base_[m].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
//	//			// Compute hash code.
//	//			auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
//	//			hash_code = stl::dynamic_bitset(descriptionDimension);
//	//			for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
//	//			{
//	//				hash_code[j] = hash_base_array_CPU[(i*(myCascadeHasher.nb_hash_code_) + j)][m] > 0;
//	//			}
//
//	//			// Determine the bucket index for each group.
//	//			//Eigen::VectorXf secondary_projection;
//	//			for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
//	//			{
//	//				uint16_t bucket_id = 0;
//
//	//				float *secondary_projection_CPU;
//	//				float *secondary_projection_GPU;
//	//				int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
//	//				cudaMalloc((void **)secondary_projection_GPU,
//	//					sizeof(float) * secondary_projection_CPU_size);
//	//				//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
//	//				myCascadeHasher.determine_buket_index_for_each_group(
//	//					secondary_projection_GPU,
//	//					secondary_hash_projection_data_GPU[j],
//	//					mat_I_point_array_GPU[m],
//	//					myCascadeHasher.nb_bits_per_bucket_,
//	//					myCascadeHasher.nb_hash_code_,
//	//					mat_I_cols[m]
//	//				);
//	//				cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
//	//					sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
//
//	//				Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
//	//					(secondary_projection_CPU, secondary_projection_CPU_size);
//
//	//				for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
//	//				{
//	//					bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
//	//				}
//	//				hashed_base_[m].hashed_desc[i].bucket_ids[j] = bucket_id;
//	//			}
//	//		}
//
//	//		//free
//	//		{
//	//			cudaFree(mat_I_point_array_GPU[m]);
//	//			mat_I_point_array_GPU[m] = NULL;
//	//			hash_base_array_CPU[m] = NULL;
//	//		}
//
//	//		// Build the Buckets
//	//		{
//	//			hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
//	//			for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
//	//			{
//	//				hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);
//
//	//				// Add the descriptor ID to the proper bucket group and id.
//	//				for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
//	//				{
//	//					const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
//	//					hashed_base_[m].buckets[i][bucket_id].push_back(j);
//	//				}
//	//			}
//	//		}
//	//	}
//	//	//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
//	//	{
//	//		char file_io_temp_i[2] = { ' ','\0' };
//	//		file_io_temp_i[0] = secondIter + 48;
//	//		const std::string file_io_str_i = file_io_temp_i;
//
//	//		char file_name_temp[2] = { ' ','\0' };
//	//		file_name_temp[0] = m + 48;
//	//		const std::string file_name_temp_m = file_name_temp;
//	//		const std::string file_name_temp2 = "block_" + file_name_temp_m;
//	//		const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
//	//		if (!stlplus::file_exists(sHash)) {
//	//			hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
//	//		}
//	//	}
//	//}
//}
int computeMatches::computeHashes
(
	Eigen::VectorXf &_zero_mean_descriptor
)

{
	//fundamental matrix
	std::string sGeometricModel = "f";
	//lowe's filter radio
	float fDistRatio = 0.8f;
	int iMatchingVideoMode = -1;
	std::string sPredefinedPairList = "";
	std::string sNearestMatchingMethod = "AUTO";
	bool bForce = false;
	bool bGuided_matching = false;
	int imax_iteration = 2048;
	unsigned int ui_max_cache_size = 0;
	
	//当前组目录
	std::string sSfM_Data_Filename_hash;
	std::string sMatchesOutputDir_hash;
	//预读组目录
	std::string sSfM_Data_Filename_hash_pre;
	std::string sMatchesOutputDir_hash_pre;

	//当前组数据
	SfM_Data sfm_data_hash;
	//预读组数据
	SfM_Data sfm_data_hash_pre;

	std::cout << " You called : " << "\n"
		<< "computeMatches" << "\n"
		<< "--input_file " << sSfM_Data_Filename_hash << "\n"
		<< "--out_dir " << sMatchesOutputDir_hash << "\n"
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
	// none of use
	if (sPredefinedPairList.length()) {
		ePairmode = PAIR_FROM_FILE;
		if (iMatchingVideoMode>0) {
			std::cerr << "\nIncompatible options: --videoModeMatching and --pairList" << std::endl;
			return EXIT_FAILURE;
		}
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
	default:
		std::cerr << "Unknown geometric model" << std::endl;
		return EXIT_FAILURE;
	}
	//分组读入图片数据之前就先把GPU上常用的空间先申请好//需要上传的一直用的数据也都上传
	openMVG::matching::RTOC myRTOC;
	CascadeHasherGPU myCascadeHasher;
	myCascadeHasher.Init(descriptionDimension);
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////这里先上传哈希计算所需要的数据 primary_hash_projection_ 
	size_t primary_hash_projection_size = (myCascadeHasher.primary_hash_projection_.rows()) * (myCascadeHasher.primary_hash_projection_.cols());
	//cpu上存放primary的指针
	const float *primary_hash_projection_data = myCascadeHasher.primary_hash_projection_.data();
	float *primary_hash_projection_data_device;
	cudaMalloc((void **)&primary_hash_projection_data_device, sizeof(float) * primary_hash_projection_size);
	cudaMemcpy(primary_hash_projection_data_device, primary_hash_projection_data, sizeof(float) * primary_hash_projection_size, cudaMemcpyHostToDevice);
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////这里上传secondary_hash_projection_ 
	//myCascadeHasher.secondary_hash_projection_.size() = 6
	
	float *secondary_hash_projection_data_CPU[6];
	float *secondary_hash_projection_data_GPU[6];
	for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; i++) {
		secondary_hash_projection_data_CPU[i] = myCascadeHasher.secondary_hash_projection_[i].data();
		size_t secondary_hash_projection_per_size = myCascadeHasher.secondary_hash_projection_[i].rows() *
													myCascadeHasher.secondary_hash_projection_[i].cols();
		cudaMalloc((void **)&secondary_hash_projection_data_GPU[i], sizeof(float) * secondary_hash_projection_per_size);
		cudaMemcpy(secondary_hash_projection_data_GPU[i], secondary_hash_projection_data_CPU[i], 
					sizeof(float) * secondary_hash_projection_per_size, cudaMemcpyHostToDevice);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//分组读入图片数据，1个sfm_data就是一组的图像数据
	//这里先计算哈希值
	for (int firstIter = 0; firstIter < group_count; firstIter++) {
		char temp_i[2] = { ' ','\0' };
		temp_i[0] = firstIter + 48;
		const std::string str_i = temp_i;

		if (firstIter == 0) {
			sSfM_Data_Filename_hash = sSfM_Data_FilenameDir_father + "DJI_" + str_i + "_build/" + "sfm_data.json";
			sMatchesOutputDir_hash = sMatchesOutputDir_father + "DJI_" + str_i + "_build/";

			temp_i[0] = firstIter + 1 + 48;
			const std::string str_i_plus_1 = temp_i;
			sSfM_Data_Filename_hash_pre = sSfM_Data_FilenameDir_father + "DJI_" + str_i_plus_1 + "_build/" + "sfm_data.json";
			sMatchesOutputDir_hash_pre = sMatchesOutputDir_father + "DJI_" + str_i_plus_1 + "_build/";
			//先读进来两组数据，处理第一组，第一组处理完后把第二组的值赋给第一组，再从磁盘里读下一组放到pre里
			{
				
				//当前组目录
				if (sMatchesOutputDir_hash.empty() || !stlplus::is_folder(sMatchesOutputDir_hash)) {
					std::cerr << "\nIt is an invalid output directory" << std::endl;
					return EXIT_FAILURE;
				}
				//预读组目录
				if (sMatchesOutputDir_hash_pre.empty() || !stlplus::is_folder(sMatchesOutputDir_hash_pre)) {
					std::cerr << "\nIt is an invalid output directory" << std::endl;
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
				//当前图像组的数据
				//SfM_Data sfm_data;
				if (!Load(sfm_data_hash, sSfM_Data_Filename_hash, ESfM_Data(VIEWS | INTRINSICS))) {
					std::cerr << std::endl
						<< "The input SfM_Data file \"" << sSfM_Data_Filename_hash << "\" cannot be read." << std::endl;
					return EXIT_FAILURE;
				}
				//预读图像组的数据
				if (!Load(sfm_data_hash_pre, sSfM_Data_Filename_hash_pre, ESfM_Data(VIEWS | INTRINSICS))) {
					std::cerr << std::endl
						<< "The input SfM_Data file \"" << sSfM_Data_Filename_hash_pre << "\" cannot be read." << std::endl;
					return EXIT_FAILURE;
				}
				//---------------------------------------
				// 处理第一组的数据
				//---------------------------------------
				{
					//---------------------------------------
					// Load SfM Scene regions
					//---------------------------------------
					// Init the regions_type from the image describer file (used for image regions extraction)
					using namespace openMVG::features;
					const std::string sImage_describer = stlplus::create_filespec(sMatchesOutputDir_hash, "image_describer", "json");
					//The default regions_type is SIFT_Regions
					//The default SIFT_Regions is Scalar type
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

					if (!regions_provider->load(sfm_data_hash, sMatchesOutputDir_hash, regions_type, &progress)) {
						std::cerr << std::endl << "Invalid regions." << std::endl;
						return EXIT_FAILURE;
					}

					PairWiseMatches map_PutativesMatches;

					// Build some alias from SfM_Data Views data:
					// - List views as a vector of filenames & image sizes
					std::vector<std::string> vec_fileNames;
					std::vector<std::pair<size_t, size_t>> vec_imagesSize;
					{
						vec_fileNames.reserve(sfm_data_hash.GetViews().size());
						vec_imagesSize.reserve(sfm_data_hash.GetViews().size());
						for (Views::const_iterator iter = sfm_data_hash.GetViews().begin();
							iter != sfm_data_hash.GetViews().end();
							++iter)
						{
							const View * v = iter->second.get();
							vec_fileNames.push_back(stlplus::create_filespec(sfm_data_hash.s_root_path,
								v->s_Img_path));
							vec_imagesSize.push_back(std::make_pair(v->ui_width, v->ui_height));
						}
					}

					std::cout << std::endl << " - COMPUTE HASH(for 1 group) - " << std::endl;
					// If the matches already exists, reload them
					if (!bForce
						&& (stlplus::file_exists(sMatchesOutputDir_hash + "/matches.putative.txt")
							|| stlplus::file_exists(sMatchesOutputDir_hash + "/matches.putative.bin"))
						)
					{
						if (!(Load(map_PutativesMatches, sMatchesOutputDir_hash + "/matches.putative.bin") ||
							Load(map_PutativesMatches, sMatchesOutputDir_hash + "/matches.putative.txt")))
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
								//default set runs here
								std::cout << "Using FAST_CASCADE_HASHING_L2 matcher" << std::endl;
								//collectionMatcher.reset(new Cascade_Hashing_Matcher_Regions_GPU(fDistRatio));
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
							/*std::cerr << "Invalid Nearest Neighbor method: " << sNearestMatchingMethod << std::endl;
							return EXIT_FAILURE;*/
						}
						// Perform the cascade hashing
						system::Timer timer;
						{
							// Collect used view indexes for an image group
							std::set<IndexT> used_index;
							for (int i = 0; i < image_count_per_group; i++) {
								used_index.insert(firstIter * image_count_per_group + i);
							}
							//openMVG::matching_image_collection::Cascade_Hash_Generate sCascade_Hash_Generate;
							//第二层数据调度策略 CPU内存 <--> GPU内存
							//1.启用零拷贝内存
							cudaSetDeviceFlags(cudaDeviceMapHost);

							std::vector <int> mat_I_cols;
							mat_I_cols.resize(image_count_per_block);

							std::vector <int> mat_I_pre_cols;
							mat_I_pre_cols.resize(image_count_per_block);

							//host存放当前块内的图像描述符数据的指针数组
							const float *mat_I_point_array_CPU[image_count_per_block];
							//host存放当前块数据哈希计算结果的指针数组
							float *hash_base_array_CPU[image_count_per_block];
							//device存放当前块内的图像描述符数据的指针数组
							float *mat_I_point_array_GPU[image_count_per_block];
							//device存放当前块数据哈希计算结果的指针数组
							float *hash_base_array_GPU[image_count_per_block];
							//保存整个块内数据描述符的数量大小

							//host存放预读块内的图像描述符数据的指针数组
							const float *mat_I_pre_point_array_CPU[image_count_per_block];
							//host存放预读块内数据哈希计算结果的指针数组
							//float *hash_base_pre_array_CPU[image_count_per_block];
							//device存放预读块内的图像描述符数据的指针数组
							float *mat_I_pre_point_array_GPU[image_count_per_block];
							//device存放预读块数据哈希计算结果的指针数组
							//float *hash_base_pre_array_GPU[image_count_per_block];

							//initialize the pointer array
							for (int i = 0; i < image_count_per_block; i++) {
								mat_I_point_array_CPU[i] = NULL;
								hash_base_array_CPU[i] = NULL;
								mat_I_point_array_GPU[i] = NULL;
								hash_base_array_GPU[i] = NULL;

								mat_I_pre_point_array_CPU[i] = NULL;
								//hash_base_pre_array_CPU[i] = NULL;
								mat_I_pre_point_array_GPU[i] = NULL;
								//hash_base_pre_array_GPU[i] = NULL;
							}

							for (int secondIter = 0; secondIter < block_count_per_group; secondIter++) {
								//处理每一块的数据，保证大概能把GPU内存塞满(或许应该多次试验看哪一组实验效果最好)
								
								{
									if (secondIter == 0) {
										//store all hash result for this block
										std::map<IndexT, HashedDescriptions> hashed_base_;
										//处理当前块内每一张图片的数据后上传到GPU上
										for (int m = 0; m < image_count_per_block; ++m)
										{
											std::set<IndexT>::const_iterator iter = used_index.begin();
											std::advance(iter, m+secondIter*image_count_per_block);
											const IndexT I = *iter;
											const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
											const unsigned char * tabI =
												reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
											const size_t dimension = regionsI->DescriptorLength();
											Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
											//descriptions minus zero_mean_descriptors before upload 
											Eigen::MatrixXf descriptionsMat;
											descriptionsMat = mat_I.template cast<float>();
											for (int k = 0; k < descriptionsMat.cols(); k++) {
												descriptionsMat.col(k) -= _zero_mean_descriptor;
											}
											
											float *descriptionsMat_data_temp = descriptionsMat.data();
											mat_I_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
											//upload mat_I_array_CPU[m]
											{
												size_t descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
												cudaMalloc((void **)&mat_I_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
												cudaMemcpy(mat_I_point_array_GPU[m], mat_I_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
											}
											mat_I_cols[m] = mat_I.cols();
											
										}

										//处理预读块内每一张图片的数据后上传到GPU上
										for (int m = 0; m < image_count_per_block; ++m)
										{
											std::set<IndexT>::const_iterator iter = used_index.begin();
											std::advance(iter, m + (secondIter+1)*image_count_per_block);
											const IndexT I = *iter;
											const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
											const unsigned char * tabI =
												reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
											const size_t dimension = regionsI->DescriptorLength();
											Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
											//descriptions minus zero_mean_descriptors before upload 
											Eigen::MatrixXf descriptionsMat;
											descriptionsMat = mat_I.template cast<float>();
											for (int k = 0; k < descriptionsMat.cols(); k++) {
												descriptionsMat.col(k) -= _zero_mean_descriptor;
											}
											float *descriptionsMat_data_temp = descriptionsMat.data();
											mat_I_pre_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
											//upload mat_I_pre_array_CPU[m]
											{
												size_t descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
												cudaMalloc((void **)&mat_I_pre_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
												cudaMemcpy(mat_I_pre_point_array_GPU[m], mat_I_pre_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
											}
											mat_I_pre_cols[m] = mat_I.cols();
										}
										
										//为当前块数据做hash
										{
											for (int m = 0; m < image_count_per_block; ++m) {
												size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
												cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
												hash_base_array_GPU[m] = myCascadeHasher.hash_gen( mat_I_cols[m], descriptionDimension,
													primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
												hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
												cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);
												

												//free
												{
													mat_I_point_array_CPU[m] = NULL;


													cudaFree(hash_base_array_GPU[m]);
													hash_base_array_GPU[m] = NULL;
												}

												//Determine the bucket index for each IMG in a block.
												float *secondary_projection_CPU = NULL;
												float *secondary_projection_GPU = NULL;
												int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
												secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
												cudaMalloc((void **)secondary_projection_GPU,
													sizeof(float) * secondary_projection_CPU_size);
												//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
												secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
													//左式
													secondary_hash_projection_data_GPU[firstIter],
													//右式
													mat_I_point_array_GPU[m],
													//左式A行数
													myCascadeHasher.nb_bits_per_bucket_,
													//左式A列数
													myCascadeHasher.nb_hash_code_,
													//右式B列数
													mat_I_cols[m]
												);
												//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
												//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
												cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
													sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
												cudaFree(secondary_projection_GPU);
												//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;

												
												//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
													//(secondary_projection_CPU, secondary_projection_CPU_size);
												//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
													//(secondary_projection_CPU, secondary_projection_CPU_size);

												//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面
												
												{
													int imgCountBeforeBlockM = 0;
													for (int sss = 0; sss < m; sss++) {
														imgCountBeforeBlockM += mat_I_cols[m];
													}
													for (int i = 0; i < mat_I_cols[m]; ++i) {
														// Allocate space for each bucket id.
														IndexT m_index = m;
														hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
														hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
														// Compute hash code.
														auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
														hash_code = stl::dynamic_bitset(descriptionDimension);
														for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
														{
															hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
														}

														// Determine the bucket index for each group.
														//Eigen::VectorXf secondary_projection;
														for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
														{
															uint16_t bucket_id = 0;
															for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
															{
																//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
																bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
																//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
															}
															hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
														}
													}

													//free
													{
														cudaFree(mat_I_point_array_GPU[m]);
														mat_I_point_array_GPU[m] = NULL;
														free(hash_base_array_CPU[m]);
														hash_base_array_CPU[m] = NULL;
													}

													// Build the Buckets
													{
														hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
														for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
														{
															hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

															// Add the descriptor ID to the proper bucket group and id.
															for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
															{
																const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
																hashed_base_[m].buckets[i][bucket_id].push_back(j);
															}
														}
													}
												}
												free(secondary_projection_CPU);
												secondary_projection_CPU = NULL;
											}
										}
										//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
										{
											HashedDescription tempHashedDescriptionBefore = hashed_base_[0].hashed_desc[27];
											HashedDescription tempHashedDescription = hashed_base_[0].hashed_desc[28];
											HashedDescription tempHashedDescriptionAfter = hashed_base_[0].hashed_desc[29];

											char file_io_temp_i[2] = { ' ','\0' };
											file_io_temp_i[0] = secondIter + 48;
											const std::string file_io_str_i = file_io_temp_i;

											char file_name_temp[2] = { ' ','\0' };
											file_name_temp[0] = secondIter + 48;
											const std::string file_name_temp_m = file_name_temp;
											const std::string file_name_temp2 = "block_" + file_name_temp_m;
											const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
											if (!stlplus::file_exists(sHash)) {
												hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
											}
											else {
												std::cout << sHash << " already exists" << std::endl;
											}
										}
										//变换当前块与预读块的相关数据
										for (int m = 0; m < image_count_per_block; ++m) {
											mat_I_point_array_CPU[m] = mat_I_pre_point_array_CPU[m];
											mat_I_pre_point_array_CPU[m] = NULL;
											mat_I_cols[m] = mat_I_pre_cols[m];
											cudaMalloc((void**)&mat_I_point_array_GPU[m], sizeof(float) * mat_I_cols[m] *descriptionDimension);
											cudaMalloc((void**)&hash_base_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
											mat_I_point_array_GPU[m] = mat_I_pre_point_array_GPU[m];
										}
										

										//再预读一块数据进来
										//处理预读块内每一张图片的数据后上传到GPU上
										for (int m = 0; m < image_count_per_block; ++m)
										{
											std::set<IndexT>::const_iterator iter = used_index.begin();
											std::advance(iter, m + (secondIter + 2)*image_count_per_block);
											const IndexT I = *iter;
											const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
											const unsigned char * tabI =
												reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
											const size_t dimension = regionsI->DescriptorLength();
											Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
											//descriptions minus zero_mean_descriptors before upload 
											Eigen::MatrixXf descriptionsMat;
											descriptionsMat = mat_I.template cast<float>();
											for (int k = 0; k < descriptionsMat.cols(); k++) {
												descriptionsMat.col(k) -= _zero_mean_descriptor;
											}
											float *descriptionsMat_data_temp = descriptionsMat.data();
											mat_I_pre_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
											//upload mat_I_pre_array_CPU[m]
											{
												int descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
												cudaMalloc((void **)&mat_I_pre_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
												cudaMemcpy(mat_I_pre_point_array_GPU[m], mat_I_pre_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
											}
											mat_I_pre_cols[m] = mat_I.cols();
										}
									}
									else if (secondIter > 0 && secondIter < block_count_per_group - 2) {
										//store all hash result for this block
										std::map<IndexT, HashedDescriptions> hashed_base_;
										// 同步函数
										cudaThreadSynchronize();
										//为当前块数据做hash
										{
											for (int m = 0; m < image_count_per_block; ++m) {
												size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
												cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
												hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
													primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
												hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
												cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


												//free
												{
													cudaFree(hash_base_array_GPU[m]);
													mat_I_point_array_CPU[m] = NULL;
													hash_base_array_GPU[m] = NULL;
												}

												//Determine the bucket index for each IMG in a block.
												float *secondary_projection_CPU = NULL;
												float *secondary_projection_GPU = NULL;
												int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
												secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
												cudaMalloc((void **)secondary_projection_GPU,
													sizeof(float) * secondary_projection_CPU_size);
												//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
												secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
													//左式
													secondary_hash_projection_data_GPU[firstIter],
													//右式
													mat_I_point_array_GPU[m],
													//左式A行数
													myCascadeHasher.nb_bits_per_bucket_,
													//左式A列数
													myCascadeHasher.nb_hash_code_,
													//右式B列数
													mat_I_cols[m]
												);
												//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
												//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
												cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
													sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
												//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


												//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
												//(secondary_projection_CPU, secondary_projection_CPU_size);
												//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
												//(secondary_projection_CPU, secondary_projection_CPU_size);

												//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

												{
													int imgCountBeforeBlockM = 0;
													for (int sss = 0; sss < m; sss++) {
														imgCountBeforeBlockM += mat_I_cols[m];
													}
													for (int i = 0; i < mat_I_cols[m]; ++i) {
														//std::cout << "i: "<< i << std::endl;
														// Allocate space for each bucket id.
														IndexT m_index = m;
														hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
														hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
														// Compute hash code.
														auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
														hash_code = stl::dynamic_bitset(descriptionDimension);
														for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
														{
															hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
														}

														// Determine the bucket index for each group.
														//Eigen::VectorXf secondary_projection;
														for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
														{
															//std::cout << "j: " << j << std::endl;
															uint16_t bucket_id = 0;



															for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
															{
																//std::cout << "k: " << k << std::endl;
																//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
																bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
																//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
															}
															hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
														}
													}

													//free
													{
														cudaFree(mat_I_point_array_GPU[m]);
														mat_I_point_array_GPU[m] = NULL;
														free(hash_base_array_CPU[m]);
														hash_base_array_CPU[m] = NULL;
													}

													// Build the Buckets
													{
														hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
														for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
														{
															hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

															// Add the descriptor ID to the proper bucket group and id.
															for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
															{
																const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
																hashed_base_[m].buckets[i][bucket_id].push_back(j);
															}
														}
													}
												}
												free(secondary_projection_CPU);
												secondary_projection_CPU = NULL;
											}
										}

										//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
										{
											char file_io_temp_i[2] = { ' ','\0' };
											file_io_temp_i[0] = secondIter + 48;
											const std::string file_io_str_i = file_io_temp_i;

											char file_name_temp[2] = { ' ','\0' };
											file_name_temp[0] = secondIter + 48;
											const std::string file_name_temp_m = file_name_temp;
											const std::string file_name_temp2 = "block_" + file_name_temp_m;
											const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
											if (!stlplus::file_exists(sHash)) {
												hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
											}
											else {
												std::cout << sHash << " already exists" << std::endl;
											}
										}
										//变换当前块与预读块的相关数据
										for (int m = 0; m < image_count_per_block; ++m) {
											mat_I_point_array_CPU[m] = mat_I_pre_point_array_CPU[m];
											mat_I_pre_point_array_CPU[m] = NULL;
											mat_I_cols[m] = mat_I_pre_cols[m];
											cudaMalloc((void**)&mat_I_point_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
											cudaMalloc((void**)&hash_base_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
											mat_I_point_array_GPU[m] = mat_I_pre_point_array_GPU[m];
										}
										//再预读一块数据进来
										//处理预读块内每一张图片的数据后上传到GPU上
										for (int m = 0; m < image_count_per_block; ++m)
										{
											std::set<IndexT>::const_iterator iter = used_index.begin();
											std::advance(iter, m + (secondIter + 2)*image_count_per_block);
											const IndexT I = *iter;
											const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
											const unsigned char * tabI =
												reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
											const size_t dimension = regionsI->DescriptorLength();

											Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
											//descriptions minus zero_mean_descriptors before upload 
											Eigen::MatrixXf descriptionsMat;
											descriptionsMat = mat_I.template cast<float>();
											for (int k = 0; k < descriptionsMat.cols(); k++) {
												descriptionsMat.col(k) -= _zero_mean_descriptor;
											}
											float *descriptionsMat_data_temp = descriptionsMat.data();
											mat_I_pre_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
											//upload mat_I_pre_array_CPU[m]
											{
												int descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
												cudaMalloc((void **)&mat_I_pre_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
												cudaMemcpy(mat_I_pre_point_array_GPU[m], mat_I_pre_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
											}
											mat_I_pre_cols[m] = mat_I.cols();
										}
									}
									else if (secondIter == block_count_per_group - 2) {
										//store all hash result for this block
										std::map<IndexT, HashedDescriptions> hashed_base_;
										// 同步函数
										cudaThreadSynchronize();
										//为当前块数据做hash
										{
											for (int m = 0; m < image_count_per_block; ++m) {
												size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
												cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
												hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
													primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
												hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
												cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


												//free
												{
													cudaFree(hash_base_array_GPU[m]);
													mat_I_point_array_CPU[m] = NULL;
													hash_base_array_GPU[m] = NULL;
												}

												//Determine the bucket index for each IMG in a block.
												float *secondary_projection_CPU = NULL;
												float *secondary_projection_GPU = NULL;
												int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
												secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
												cudaMalloc((void **)secondary_projection_GPU,
													sizeof(float) * secondary_projection_CPU_size);
												//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
												secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
													//左式
													secondary_hash_projection_data_GPU[firstIter],
													//右式
													mat_I_point_array_GPU[m],
													//左式A行数
													myCascadeHasher.nb_bits_per_bucket_,
													//左式A列数
													myCascadeHasher.nb_hash_code_,
													//右式B列数
													mat_I_cols[m]
												);
												//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
												//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
												cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
													sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
												//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


												//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
												//(secondary_projection_CPU, secondary_projection_CPU_size);
												//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
												//(secondary_projection_CPU, secondary_projection_CPU_size);

												//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

												{
													int imgCountBeforeBlockM = 0;
													for (int sss = 0; sss < m; sss++) {
														imgCountBeforeBlockM += mat_I_cols[m];
													}
													for (int i = 0; i < mat_I_cols[m]; ++i) {
														// Allocate space for each bucket id.
														IndexT m_index = m;
														hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
														hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
														// Compute hash code.
														auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
														hash_code = stl::dynamic_bitset(descriptionDimension);
														for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
														{
															hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
														}

														// Determine the bucket index for each group.
														//Eigen::VectorXf secondary_projection;
														for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
														{
															uint16_t bucket_id = 0;



															for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
															{
																//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
																bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
																//bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((imgCountBeforeBlockM + i)*(myCascadeHasher.nb_hash_code_) + k)] > 0 ? 1 : 0);
																//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
															}
															hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
														}
													}

													//free
													{
														cudaFree(mat_I_point_array_GPU[m]);
														mat_I_point_array_GPU[m] = NULL;
														free(hash_base_array_CPU[m]);
														hash_base_array_CPU[m] = NULL;
													}

													// Build the Buckets
													{
														hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
														for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
														{
															hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

															// Add the descriptor ID to the proper bucket group and id.
															for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
															{
																const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
																hashed_base_[m].buckets[i][bucket_id].push_back(j);
															}
														}
													}
												}
												free(secondary_projection_CPU);
												secondary_projection_CPU = NULL;
											}
										}
										//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
										{
											char file_io_temp_i[2] = { ' ','\0' };
											file_io_temp_i[0] = secondIter + 48;
											const std::string file_io_str_i = file_io_temp_i;

											char file_name_temp[2] = { ' ','\0' };
											file_name_temp[0] = secondIter + 48;
											const std::string file_name_temp_m = file_name_temp;
											const std::string file_name_temp2 = "block_" + file_name_temp_m;
											const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
											if (!stlplus::file_exists(sHash)) {
												hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
											}
											else {
												std::cout << sHash << " already exists" << std::endl;
											}
										}
										//变换当前块与预读块的相关数据
										secondIter++;
										for (int m = 0; m < image_count_per_block; ++m) {
											mat_I_point_array_CPU[m] = mat_I_pre_point_array_CPU[m];
											mat_I_pre_point_array_CPU[m] = NULL;
											mat_I_cols[m] = mat_I_pre_cols[m];
											cudaMalloc((void**)&mat_I_point_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
											cudaMalloc((void**)&hash_base_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
											mat_I_point_array_GPU[m] = mat_I_pre_point_array_GPU[m];
										}
										//处理最后一块数据
										// 同步函数
										cudaThreadSynchronize();
										//为最后一块(当前的预读块)数据做hash
										{
											for (int m = 0; m < image_count_per_block; ++m) {
												size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
												cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
												hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
													primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
												hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
												cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


												//free
												{
													cudaFree(hash_base_array_GPU[m]);
													mat_I_point_array_CPU[m] = NULL;
													hash_base_array_GPU[m] = NULL;
												}

												//Determine the bucket index for each IMG in a block.
												float *secondary_projection_CPU = NULL;
												float *secondary_projection_GPU = NULL;
												int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
												secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
												cudaMalloc((void **)secondary_projection_GPU,
													sizeof(float) * secondary_projection_CPU_size);
												//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
												secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
													//左式
													secondary_hash_projection_data_GPU[firstIter],
													//右式
													mat_I_point_array_GPU[m],
													//左式A行数
													myCascadeHasher.nb_bits_per_bucket_,
													//左式A列数
													myCascadeHasher.nb_hash_code_,
													//右式B列数
													mat_I_cols[m]
												);
												//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
												//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
												cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
													sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
												//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


												//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
												//(secondary_projection_CPU, secondary_projection_CPU_size);
												//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
												//(secondary_projection_CPU, secondary_projection_CPU_size);

												//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

												{
													int imgCountBeforeBlockM = 0;
													for (int sss = 0; sss < m; sss++) {
														imgCountBeforeBlockM += mat_I_cols[m];
													}
													for (int i = 0; i < mat_I_cols[m]; ++i) {
														// Allocate space for each bucket id.
														IndexT m_index = m;
														hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
														hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
														// Compute hash code.
														auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
														hash_code = stl::dynamic_bitset(descriptionDimension);
														for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
														{
															hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
														}

														// Determine the bucket index for each group.
														//Eigen::VectorXf secondary_projection;
														for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
														{
															uint16_t bucket_id = 0;



															for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
															{
																//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
																//bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((imgCountBeforeBlockM + i)*(myCascadeHasher.nb_hash_code_) + k)] > 0 ? 1 : 0);
																bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
																//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
															}
															hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
														}
													}

													//free
													{
														cudaFree(mat_I_point_array_GPU[m]);
														mat_I_point_array_GPU[m] = NULL;
														free(hash_base_array_CPU[m]);
														hash_base_array_CPU[m] = NULL;
													}

													// Build the Buckets
													{
														hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
														for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
														{
															hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

															// Add the descriptor ID to the proper bucket group and id.
															for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
															{
																const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
																hashed_base_[m].buckets[i][bucket_id].push_back(j);
															}
														}
													}
												}
												free(secondary_projection_CPU);
												secondary_projection_CPU = NULL;
											}
										}
										//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
										{
											char file_io_temp_i[2] = { ' ','\0' };
											file_io_temp_i[0] = secondIter + 48;
											const std::string file_io_str_i = file_io_temp_i;

											char file_name_temp[2] = { ' ','\0' };
											file_name_temp[0] = secondIter + 48;
											const std::string file_name_temp_m = file_name_temp;
											const std::string file_name_temp2 = "block_" + file_name_temp_m;
											const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
											if (!stlplus::file_exists(sHash)) {
												hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
											}
											else {
												std::cout << sHash << " already exists" << std::endl;
											}
										}
									}
									else {
										std::cerr << "error when index the secondIter!:"<< secondIter << std::endl;
										return EXIT_FAILURE;
									}
								}
							}
						}
						std::cout << "Task (Regions Hashing for group " <<firstIter << ") done in (s): " << timer.elapsed() << std::endl;
					}
				}
				//
				//---------------------------------------
				// 把pre赋值给当前组数据，并读入第三组的数据作为pre
				//---------------------------------------
				sSfM_Data_Filename_hash = sSfM_Data_Filename_hash_pre;
				sMatchesOutputDir_hash = sMatchesOutputDir_hash_pre;
				sfm_data_hash = sfm_data_hash_pre;

				temp_i[0] = firstIter + 2 + 48;
				const std::string str_i_plus_2 = temp_i;
				sSfM_Data_Filename_hash_pre = sSfM_Data_FilenameDir_father + "DJI_" + str_i_plus_2 + "_build/" + "sfm_data.json";
				sMatchesOutputDir_hash_pre = sMatchesOutputDir_father + "DJI_" + str_i_plus_2 + "_build/";
				
				//预读图像组的数据
				if (!Load(sfm_data_hash_pre, sSfM_Data_Filename_hash_pre, ESfM_Data(VIEWS | INTRINSICS))) {
					std::cerr << std::endl
						<< "The input SfM_Data file \"" << sSfM_Data_Filename_hash_pre << "\" cannot be read." << std::endl;
					return EXIT_FAILURE;
				}
				
				//return EXIT_SUCCESS;
			}
		}
		else if(firstIter>0 && firstIter<group_count-2)
		{
			//---------------------------------------
			// 直接处理当前组的数据
			//---------------------------------------
			 {
				//---------------------------------------
				// Load SfM Scene regions
				//---------------------------------------
				// Init the regions_type from the image describer file (used for image regions extraction)
				using namespace openMVG::features;
				const std::string sImage_describer = stlplus::create_filespec(sMatchesOutputDir_hash, "image_describer", "json");
				//The default regions_type is SIFT_Regions
				//The default SIFT_Regions is Scalar type
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

				if (!regions_provider->load(sfm_data_hash, sMatchesOutputDir_hash, regions_type, &progress)) {
					std::cerr << std::endl << "Invalid regions." << std::endl;
					return EXIT_FAILURE;
				}

				PairWiseMatches map_PutativesMatches;

				// Build some alias from SfM_Data Views data:
				// - List views as a vector of filenames & image sizes
				std::vector<std::string> vec_fileNames;
				std::vector<std::pair<size_t, size_t>> vec_imagesSize;
				{
					vec_fileNames.reserve(sfm_data_hash.GetViews().size());
					vec_imagesSize.reserve(sfm_data_hash.GetViews().size());
					for (Views::const_iterator iter = sfm_data_hash.GetViews().begin();
						iter != sfm_data_hash.GetViews().end();
						++iter)
					{
						const View * v = iter->second.get();
						vec_fileNames.push_back(stlplus::create_filespec(sfm_data_hash.s_root_path,
							v->s_Img_path));
						vec_imagesSize.push_back(std::make_pair(v->ui_width, v->ui_height));
					}
				}

				std::cout << std::endl << " - COMPUTE HASH(for 1 group) - " << std::endl;
				// If the matches already exists, reload them
				if (!bForce
					&& (stlplus::file_exists(sMatchesOutputDir_hash + "/matches.putative.txt")
						|| stlplus::file_exists(sMatchesOutputDir_hash + "/matches.putative.bin"))
					)
				{
					if (!(Load(map_PutativesMatches, sMatchesOutputDir_hash + "/matches.putative.bin") ||
						Load(map_PutativesMatches, sMatchesOutputDir_hash + "/matches.putative.txt")))
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
							//default set runs here
							std::cout << "Using FAST_CASCADE_HASHING_L2 matcher" << std::endl;
							//collectionMatcher.reset(new Cascade_Hashing_Matcher_Regions_GPU(fDistRatio));
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
						//std::cerr << "Invalid Nearest Neighbor method: " << sNearestMatchingMethod << std::endl;
						//return EXIT_FAILURE;
					}
					// Perform the cascade hashing
					system::Timer timer;
					{
						// Collect used view indexes for an image group
						std::set<IndexT> used_index;
						for (int i = 0; i < image_count_per_group; i++) {
							used_index.insert(i);
							//used_index.insert(firstIter * image_count_per_group + i);
						}
						//openMVG::matching_image_collection::Cascade_Hash_Generate sCascade_Hash_Generate;
						//第二层数据调度策略 CPU内存 <--> GPU内存
						//1.启用零拷贝内存
						cudaSetDeviceFlags(cudaDeviceMapHost);

						std::vector <int> mat_I_cols;
						mat_I_cols.resize(image_count_per_block);

						std::vector <int> mat_I_pre_cols;
						mat_I_pre_cols.resize(image_count_per_block);

						//host存放当前块内的图像描述符数据的指针数组
						const float *mat_I_point_array_CPU[image_count_per_block];
						//host存放当前块数据哈希计算结果的指针数组
						float *hash_base_array_CPU[image_count_per_block];
						//device存放当前块内的图像描述符数据的指针数组
						float *mat_I_point_array_GPU[image_count_per_block];
						//device存放当前块数据哈希计算结果的指针数组
						float *hash_base_array_GPU[image_count_per_block];
						//保存整个块内数据描述符的数量大小

						//host存放预读块内的图像描述符数据的指针数组
						const float *mat_I_pre_point_array_CPU[image_count_per_block];
						//host存放预读块内数据哈希计算结果的指针数组
						//float *hash_base_pre_array_CPU[image_count_per_block];
						//device存放预读块内的图像描述符数据的指针数组
						float *mat_I_pre_point_array_GPU[image_count_per_block];
						//device存放预读块数据哈希计算结果的指针数组
						//float *hash_base_pre_array_GPU[image_count_per_block];

						//initialize all pointers
						for (int i = 0; i < image_count_per_block; i++) {
							mat_I_point_array_CPU[i] = NULL;
							hash_base_array_CPU[i] = NULL;
							mat_I_point_array_GPU[i] = NULL;
							hash_base_array_GPU[i] = NULL;

							mat_I_pre_point_array_CPU[i] = NULL;
							//hash_base_pre_array_CPU[i] = NULL;
							mat_I_pre_point_array_GPU[i] = NULL;
							//hash_base_pre_array_GPU[i] = NULL;
						}

						for (int secondIter = 0; secondIter < block_count_per_group; secondIter++) {
							//处理每一块的数据，保证大概能把GPU内存塞满(或许应该多次试验看哪一组实验效果最好)
							{
								if (secondIter == 0) {
									//store all hash result for this block
									std::map<IndexT, HashedDescriptions> hashed_base_;
									//处理当前块内每一张图片的数据后上传到GPU上
									for (int m = 0; m < image_count_per_block; ++m)
									{
										std::set<IndexT>::const_iterator iter = used_index.begin();
										std::advance(iter, m + secondIter*image_count_per_block);
										const IndexT I = *iter;
										const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
										const unsigned char * tabI =
											reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
										const size_t dimension = regionsI->DescriptorLength();

										Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
										//descriptions minus zero_mean_descriptors before upload 
										Eigen::MatrixXf descriptionsMat;
										descriptionsMat = mat_I.template cast<float>();
										for (int k = 0; k < descriptionsMat.cols(); k++) {
											descriptionsMat.col(k) -= _zero_mean_descriptor;
										}
										float *descriptionsMat_data_temp = descriptionsMat.data();
										mat_I_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
										//upload mat_I_array_CPU[m]
										{
											int descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
											cudaMalloc((void **)&mat_I_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
											cudaMemcpy(mat_I_point_array_GPU[m], mat_I_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
										}
										mat_I_cols[m] = mat_I.cols();

									}
									//处理预读块内每一张图片的数据后上传到GPU上
									for (int m = 0; m < image_count_per_block; ++m)
									{
										std::set<IndexT>::const_iterator iter = used_index.begin();
										std::advance(iter, m + (secondIter + 1)*image_count_per_block);
										const IndexT I = *iter;
										const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
										const unsigned char * tabI =
											reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
										const size_t dimension = regionsI->DescriptorLength();

										Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
										//descriptions minus zero_mean_descriptors before upload 
										Eigen::MatrixXf descriptionsMat;
										descriptionsMat = mat_I.template cast<float>();
										for (int k = 0; k < descriptionsMat.cols(); k++) {
											descriptionsMat.col(k) -= _zero_mean_descriptor;
										}
										float *descriptionsMat_data_temp = descriptionsMat.data();
										mat_I_pre_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
										//upload mat_I_pre_array_CPU[m]
										{
											int descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
											cudaMalloc((void **)&mat_I_pre_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
											cudaMemcpy(mat_I_pre_point_array_GPU[m], mat_I_pre_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
										}
										mat_I_pre_cols[m] = mat_I.cols();
									}

									//为当前块数据做hash
									{
										for (int m = 0; m < image_count_per_block; ++m) {
											size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
											cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
											hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
												primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
											hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
											cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


											//free
											{
												cudaFree(hash_base_array_GPU[m]);
												mat_I_point_array_CPU[m] = NULL;
												hash_base_array_GPU[m] = NULL;
											}

											//Determine the bucket index for each IMG in a block.
											float *secondary_projection_CPU = NULL;
											float *secondary_projection_GPU = NULL;
											int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
											secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
											cudaMalloc((void **)secondary_projection_GPU,
												sizeof(float) * secondary_projection_CPU_size);
											//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
											secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
												//左式
												secondary_hash_projection_data_GPU[firstIter],
												//右式
												mat_I_point_array_GPU[m],
												//左式A行数
												myCascadeHasher.nb_bits_per_bucket_,
												//左式A列数
												myCascadeHasher.nb_hash_code_,
												//右式B列数
												mat_I_cols[m]
											);
											//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
											//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
											cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
												sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
											//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


											//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
											//(secondary_projection_CPU, secondary_projection_CPU_size);
											//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
											//(secondary_projection_CPU, secondary_projection_CPU_size);

											//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

											{
												int imgCountBeforeBlockM = 0;
												for (int sss = 0; sss < m; sss++) {
													imgCountBeforeBlockM += mat_I_cols[m];
												}
												for (int i = 0; i < mat_I_cols[m]; ++i) {
													// Allocate space for each bucket id.
													IndexT m_index = m;
													hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
													hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
													// Compute hash code.
													auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
													hash_code = stl::dynamic_bitset(descriptionDimension);
													for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
													{
														hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
													}

													// Determine the bucket index for each group.
													//Eigen::VectorXf secondary_projection;
													for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
													{
														uint16_t bucket_id = 0;



														for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
														{
															//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);

															bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((imgCountBeforeBlockM + i)*(myCascadeHasher.nb_hash_code_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
														}
														hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
													}
												}

												//free
												{
													cudaFree(mat_I_point_array_GPU[m]);
													mat_I_point_array_GPU[m] = NULL;
													free(hash_base_array_CPU[m]);
													hash_base_array_CPU[m] = NULL;
												}

												// Build the Buckets
												{
													hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
													for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
													{
														hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

														// Add the descriptor ID to the proper bucket group and id.
														for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
														{
															const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
															hashed_base_[m].buckets[i][bucket_id].push_back(j);
														}
													}
												}
											}
											free(secondary_projection_CPU);
											secondary_projection_CPU = NULL;
										}
									}
									//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
									{
										char file_io_temp_i[2] = { ' ','\0' };
										file_io_temp_i[0] = secondIter + 48;
										const std::string file_io_str_i = file_io_temp_i;

										char file_name_temp[2] = { ' ','\0' };
										file_name_temp[0] = secondIter + 48;
										const std::string file_name_temp_m = file_name_temp;
										const std::string file_name_temp2 = "block_" + file_name_temp_m;
										const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
										if (!stlplus::file_exists(sHash)) {
											hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
										}
										else {
											std::cout << sHash << " already exists" << std::endl;
										}
									}
									//变换当前块与预读块的相关数据
									for (int m = 0; m < image_count_per_block; ++m) {
										mat_I_point_array_CPU[m] = mat_I_pre_point_array_CPU[m];
										mat_I_pre_point_array_CPU[m] = NULL;
										mat_I_cols[m] = mat_I_pre_cols[m];
										cudaMalloc((void**)&mat_I_point_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										cudaMalloc((void**)&hash_base_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										mat_I_point_array_GPU[m] = mat_I_pre_point_array_GPU[m];
									}


									//再预读一块数据进来
									//处理预读块内每一张图片的数据后上传到GPU上
									for (int m = 0; m < image_count_per_block; ++m)
									{
										std::set<IndexT>::const_iterator iter = used_index.begin();
										std::advance(iter, m + (secondIter + 2)*image_count_per_block);
										const IndexT I = *iter;
										const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
										const unsigned char * tabI =
											reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
										const size_t dimension = regionsI->DescriptorLength();

										Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
										//descriptions minus zero_mean_descriptors before upload 
										Eigen::MatrixXf descriptionsMat;
										descriptionsMat = mat_I.template cast<float>();
										for (int k = 0; k < descriptionsMat.cols(); k++) {
											descriptionsMat.col(k) -= _zero_mean_descriptor;
										}
										float *descriptionsMat_data_temp = descriptionsMat.data();
										mat_I_pre_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
										//upload mat_I_pre_array_CPU[m]
										{
											int descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
											cudaMalloc((void **)&mat_I_pre_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
											cudaMemcpy(mat_I_pre_point_array_GPU[m], mat_I_pre_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
										}
										mat_I_pre_cols[m] = mat_I.cols();
									}
								}
								else if (secondIter > 0 && secondIter < block_count_per_group - 2) {
									//store all hash result for this block
									std::map<IndexT, HashedDescriptions> hashed_base_;
									// 同步函数
									cudaThreadSynchronize();
									//为当前块数据做hash
									{
										for (int m = 0; m < image_count_per_block; ++m) {
											size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
											cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
											hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
												primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
											hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
											cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


											//free
											{
												cudaFree(hash_base_array_GPU[m]);
												mat_I_point_array_CPU[m] = NULL;
												hash_base_array_GPU[m] = NULL;
											}

											//Determine the bucket index for each IMG in a block.
											float *secondary_projection_CPU = NULL;
											float *secondary_projection_GPU = NULL;
											int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
											secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
											cudaMalloc((void **)secondary_projection_GPU,
												sizeof(float) * secondary_projection_CPU_size);
											//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
											secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
												//左式
												secondary_hash_projection_data_GPU[firstIter],
												//右式
												mat_I_point_array_GPU[m],
												//左式A行数
												myCascadeHasher.nb_bits_per_bucket_,
												//左式A列数
												myCascadeHasher.nb_hash_code_,
												//右式B列数
												mat_I_cols[m]
											);
											//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
											//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
											cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
												sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
											//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


											//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
											//(secondary_projection_CPU, secondary_projection_CPU_size);
											//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
											//(secondary_projection_CPU, secondary_projection_CPU_size);

											//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

											{
												int imgCountBeforeBlockM = 0;
												for (int sss = 0; sss < m; sss++) {
													imgCountBeforeBlockM += mat_I_cols[m];
												}
												for (int i = 0; i < mat_I_cols[m]; ++i) {
													// Allocate space for each bucket id.
													IndexT m_index = m;
													hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
													hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
													// Compute hash code.
													auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
													hash_code = stl::dynamic_bitset(descriptionDimension);
													for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
													{
														hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
													}

													// Determine the bucket index for each group.
													//Eigen::VectorXf secondary_projection;
													for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
													{
														uint16_t bucket_id = 0;



														for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
														{
															//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
															bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((imgCountBeforeBlockM + i)*(myCascadeHasher.nb_hash_code_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
														}
														hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
													}
												}

												//free
												{
													cudaFree(mat_I_point_array_GPU[m]);
													mat_I_point_array_GPU[m] = NULL;
													free(hash_base_array_CPU[m]);
													hash_base_array_CPU[m] = NULL;
												}

												// Build the Buckets
												{
													hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
													for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
													{
														hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

														// Add the descriptor ID to the proper bucket group and id.
														for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
														{
															const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
															hashed_base_[m].buckets[i][bucket_id].push_back(j);
														}
													}
												}
											}
											free(secondary_projection_CPU);
											secondary_projection_CPU = NULL;
										}
									}
									//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
									{
										char file_io_temp_i[2] = { ' ','\0' };
										file_io_temp_i[0] = secondIter + 48;
										const std::string file_io_str_i = file_io_temp_i;

										char file_name_temp[2] = { ' ','\0' };
										file_name_temp[0] = secondIter + 48;
										const std::string file_name_temp_m = file_name_temp;
										const std::string file_name_temp2 = "block_" + file_name_temp_m;
										const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
										if (!stlplus::file_exists(sHash)) {
											hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
										}
										else {
											std::cout << sHash << " already exists" << std::endl;
										}
									}
									//变换当前块与预读块的相关数据
									for (int m = 0; m < image_count_per_block; ++m) {
										mat_I_point_array_CPU[m] = mat_I_pre_point_array_CPU[m];
										mat_I_pre_point_array_CPU[m] = NULL;
										mat_I_cols[m] = mat_I_pre_cols[m];
										cudaMalloc((void**)&mat_I_point_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										cudaMalloc((void**)&hash_base_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										mat_I_point_array_GPU[m] = mat_I_pre_point_array_GPU[m];
									}
									//再预读一块数据进来
									//处理预读块内每一张图片的数据后上传到GPU上
									for (int m = 0; m < image_count_per_block; ++m)
									{
										std::set<IndexT>::const_iterator iter = used_index.begin();
										std::advance(iter, m + (secondIter + 2)*image_count_per_block);
										const IndexT I = *iter;
										const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
										const unsigned char * tabI =
											reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
										const size_t dimension = regionsI->DescriptorLength();

										Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
										//descriptions minus zero_mean_descriptors before upload 
										Eigen::MatrixXf descriptionsMat;
										descriptionsMat = mat_I.template cast<float>();
										for (int k = 0; k < descriptionsMat.cols(); k++) {
											descriptionsMat.col(k) -= _zero_mean_descriptor;
										}
										float *descriptionsMat_data_temp = descriptionsMat.data();
										mat_I_pre_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
										//upload mat_I_pre_array_CPU[m]
										{
											int descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
											cudaMalloc((void **)&mat_I_pre_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
											cudaMemcpy(mat_I_pre_point_array_GPU[m], mat_I_pre_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
										}
										mat_I_pre_cols[m] = mat_I.cols();
									}
								}
								else if (secondIter == block_count_per_group - 2) {
									//store all hash result for this block
									std::map<IndexT, HashedDescriptions> hashed_base_;
									// 同步函数
									cudaThreadSynchronize();
									//为当前块数据做hash
									{
										for (int m = 0; m < image_count_per_block; ++m) {
											size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
											cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
											hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
												primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
											hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
											cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


											//free
											{
												cudaFree(hash_base_array_GPU[m]);
												mat_I_point_array_CPU[m] = NULL;
												hash_base_array_GPU[m] = NULL;
											}

											//Determine the bucket index for each IMG in a block.
											float *secondary_projection_CPU = NULL;
											float *secondary_projection_GPU = NULL;
											int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
											secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
											cudaMalloc((void **)secondary_projection_GPU,
												sizeof(float) * secondary_projection_CPU_size);
											//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
											secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
												//左式
												secondary_hash_projection_data_GPU[firstIter],
												//右式
												mat_I_point_array_GPU[m],
												//左式A行数
												myCascadeHasher.nb_bits_per_bucket_,
												//左式A列数
												myCascadeHasher.nb_hash_code_,
												//右式B列数
												mat_I_cols[m]
											);
											//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
											//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
											cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
												sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
											//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


											//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
											//(secondary_projection_CPU, secondary_projection_CPU_size);
											//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
											//(secondary_projection_CPU, secondary_projection_CPU_size);

											//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

											{
												int imgCountBeforeBlockM = 0;
												for (int sss = 0; sss < m; sss++) {
													imgCountBeforeBlockM += mat_I_cols[m];
												}
												for (int i = 0; i < mat_I_cols[m]; ++i) {
													// Allocate space for each bucket id.
													IndexT m_index = m;
													hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
													hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
													// Compute hash code.
													auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
													hash_code = stl::dynamic_bitset(descriptionDimension);
													for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
													{
														hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
													}

													// Determine the bucket index for each group.
													//Eigen::VectorXf secondary_projection;
													for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
													{
														uint16_t bucket_id = 0;



														for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
														{
															//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
															bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((imgCountBeforeBlockM + i)*(myCascadeHasher.nb_hash_code_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
														}
														hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
													}
												}

												//free
												{
													cudaFree(mat_I_point_array_GPU[m]);
													mat_I_point_array_GPU[m] = NULL;
													free(hash_base_array_CPU[m]);
													hash_base_array_CPU[m] = NULL;
												}

												// Build the Buckets
												{
													hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
													for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
													{
														hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

														// Add the descriptor ID to the proper bucket group and id.
														for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
														{
															const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
															hashed_base_[m].buckets[i][bucket_id].push_back(j);
														}
													}
												}
											}
											free(secondary_projection_CPU);
											secondary_projection_CPU = NULL;
										}
									}
									//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
									{
										char file_io_temp_i[2] = { ' ','\0' };
										file_io_temp_i[0] = secondIter + 48;
										const std::string file_io_str_i = file_io_temp_i;

										char file_name_temp[2] = { ' ','\0' };
										file_name_temp[0] = secondIter + 48;
										const std::string file_name_temp_m = file_name_temp;
										const std::string file_name_temp2 = "block_" + file_name_temp_m;
										const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
										if (!stlplus::file_exists(sHash)) {
											hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
										}
										else {
											std::cout << sHash << " already exists" << std::endl;
										}
									}
									//变换当前块与预读块的相关数据
									secondIter++;
									for (int m = 0; m < image_count_per_block; ++m) {
										mat_I_point_array_CPU[m] = mat_I_pre_point_array_CPU[m];
										mat_I_pre_point_array_CPU[m] = NULL;
										mat_I_cols[m] = mat_I_pre_cols[m];
										cudaMalloc((void**)&mat_I_point_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										cudaMalloc((void**)&hash_base_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										mat_I_point_array_GPU[m] = mat_I_pre_point_array_GPU[m];
									}
									//处理最后一块数据
									// 同步函数
									cudaThreadSynchronize();
									//为最后一块(当前的预读块)数据做hash
									{
										for (int m = 0; m < image_count_per_block; ++m) {
											size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
											cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
											hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
												primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
											hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
											cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


											//free
											{
												cudaFree(hash_base_array_GPU[m]);
												mat_I_point_array_CPU[m] = NULL;
												hash_base_array_GPU[m] = NULL;
											}

											//Determine the bucket index for each IMG in a block.
											float *secondary_projection_CPU = NULL;
											float *secondary_projection_GPU = NULL;
											int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
											secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
											cudaMalloc((void **)secondary_projection_GPU,
												sizeof(float) * secondary_projection_CPU_size);
											//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
											secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
												//左式
												secondary_hash_projection_data_GPU[firstIter],
												//右式
												mat_I_point_array_GPU[m],
												//左式A行数
												myCascadeHasher.nb_bits_per_bucket_,
												//左式A列数
												myCascadeHasher.nb_hash_code_,
												//右式B列数
												mat_I_cols[m]
											);
											//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
											//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
											cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
												sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
											//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


											//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
											//(secondary_projection_CPU, secondary_projection_CPU_size);
											//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
											//(secondary_projection_CPU, secondary_projection_CPU_size);

											//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

											{
												int imgCountBeforeBlockM = 0;
												for (int sss = 0; sss < m; sss++) {
													imgCountBeforeBlockM += mat_I_cols[m];
												}
												for (int i = 0; i < mat_I_cols[m]; ++i) {
													// Allocate space for each bucket id.
													IndexT m_index = m;
													hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
													hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
													// Compute hash code.
													auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
													hash_code = stl::dynamic_bitset(descriptionDimension);
													for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
													{
														hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
													}

													// Determine the bucket index for each group.
													//Eigen::VectorXf secondary_projection;
													for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
													{
														uint16_t bucket_id = 0;



														for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
														{
															//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
															bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((imgCountBeforeBlockM + i)*(myCascadeHasher.nb_hash_code_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
														}
														hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
													}
												}

												//free
												{
													cudaFree(mat_I_point_array_GPU[m]);
													mat_I_point_array_GPU[m] = NULL;
													free(hash_base_array_CPU[m]);
													hash_base_array_CPU[m] = NULL;
												}

												// Build the Buckets
												{
													hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
													for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
													{
														hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

														// Add the descriptor ID to the proper bucket group and id.
														for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
														{
															const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
															hashed_base_[m].buckets[i][bucket_id].push_back(j);
														}
													}
												}
											}
											free(secondary_projection_CPU);
											secondary_projection_CPU = NULL;
										}
									}
									//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
									{
										char file_io_temp_i[2] = { ' ','\0' };
										file_io_temp_i[0] = secondIter + 48;
										const std::string file_io_str_i = file_io_temp_i;

										char file_name_temp[2] = { ' ','\0' };
										file_name_temp[0] = secondIter + 48;
										const std::string file_name_temp_m = file_name_temp;
										const std::string file_name_temp2 = "block_" + file_name_temp_m;
										const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
										if (!stlplus::file_exists(sHash)) {
											hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
										}
										else {
											std::cout << sHash << " already exists" << std::endl;
										}
									}
								}
								else {
									std::cerr << "error when index the secondIter!:" << secondIter <<std::endl;
									return EXIT_FAILURE;
								}
							}
						}
					}
					std::cout << "Task (Regions Hashing for group " << firstIter << ") done in (s): " << timer.elapsed() << std::endl;
				}
			}
			///////

			//处理完当前组数据后，再执行下面几行代码，把第二组的数据赋值给第一组，再从磁盘上读取一组数据到内存里来放在pre里
			sSfM_Data_Filename_hash = sSfM_Data_Filename_hash_pre;
			sMatchesOutputDir_hash = sMatchesOutputDir_hash_pre;
			sfm_data_hash = sfm_data_hash_pre;

			temp_i[0] = firstIter + 2 + 48;
			const std::string str_i_plus_2 = temp_i;
			sSfM_Data_Filename_hash_pre = sSfM_Data_FilenameDir_father + "DJI_" + str_i_plus_2 + "_build/" + "sfm_data.json";
			sMatchesOutputDir_hash_pre = sMatchesOutputDir_father + "DJI_" + str_i_plus_2 + "_build/";

			//预读图像组的数据
			if (!Load(sfm_data_hash_pre, sSfM_Data_Filename_hash_pre, ESfM_Data(VIEWS | INTRINSICS))) {
				std::cerr << std::endl
					<< "The input SfM_Data file \"" << sSfM_Data_Filename_hash_pre << "\" cannot be read." << std::endl;
				return EXIT_FAILURE;
			}
			//return EXIT_SUCCESS;
		}
		else if (firstIter == (group_count - 2)) {
			//---------------------------------------
			// 直接处理当前组的数据
			//---------------------------------------
			{
				//---------------------------------------
				// Load SfM Scene regions
				//---------------------------------------
				// Init the regions_type from the image describer file (used for image regions extraction)
				using namespace openMVG::features;
				const std::string sImage_describer = stlplus::create_filespec(sMatchesOutputDir_hash, "image_describer", "json");
				//The default regions_type is SIFT_Regions
				//The default SIFT_Regions is Scalar type
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

				if (!regions_provider->load(sfm_data_hash, sMatchesOutputDir_hash, regions_type, &progress)) {
					std::cerr << std::endl << "Invalid regions." << std::endl;
					return EXIT_FAILURE;
				}

				PairWiseMatches map_PutativesMatches;

				// Build some alias from SfM_Data Views data:
				// - List views as a vector of filenames & image sizes
				std::vector<std::string> vec_fileNames;
				std::vector<std::pair<size_t, size_t>> vec_imagesSize;
				{
					vec_fileNames.reserve(sfm_data_hash.GetViews().size());
					vec_imagesSize.reserve(sfm_data_hash.GetViews().size());
					for (Views::const_iterator iter = sfm_data_hash.GetViews().begin();
						iter != sfm_data_hash.GetViews().end();
						++iter)
					{
						const View * v = iter->second.get();
						vec_fileNames.push_back(stlplus::create_filespec(sfm_data_hash.s_root_path,
							v->s_Img_path));
						vec_imagesSize.push_back(std::make_pair(v->ui_width, v->ui_height));
					}
				}

				std::cout << std::endl << " - COMPUTE HASH(for 1 group) - " << std::endl;
				// If the matches already exists, reload them
				if (!bForce
					&& (stlplus::file_exists(sMatchesOutputDir_hash + "/matches.putative.txt")
						|| stlplus::file_exists(sMatchesOutputDir_hash + "/matches.putative.bin"))
					)
				{
					if (!(Load(map_PutativesMatches, sMatchesOutputDir_hash + "/matches.putative.bin") ||
						Load(map_PutativesMatches, sMatchesOutputDir_hash + "/matches.putative.txt")))
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
							//default set runs here
							std::cout << "Using FAST_CASCADE_HASHING_L2 matcher" << std::endl;
							//collectionMatcher.reset(new Cascade_Hashing_Matcher_Regions_GPU(fDistRatio));
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
						//std::cerr << "Invalid Nearest Neighbor method: " << sNearestMatchingMethod << std::endl;
						//return EXIT_FAILURE;
					}
					// Perform the cascade hashing
					system::Timer timer;
					{
						// Collect used view indexes for an image group
						std::set<IndexT> used_index;
						for (int i = 0; i < image_count_per_group; i++) {
							used_index.insert(i);
							//used_index.insert(firstIter * image_count_per_group + i);
						}
						//openMVG::matching_image_collection::Cascade_Hash_Generate sCascade_Hash_Generate;
						//第二层数据调度策略 CPU内存 <--> GPU内存
						//1.启用零拷贝内存
						cudaSetDeviceFlags(cudaDeviceMapHost);

						std::vector <int> mat_I_cols;
						mat_I_cols.resize(image_count_per_block);

						std::vector <int> mat_I_pre_cols;
						mat_I_pre_cols.resize(image_count_per_block);

						//host存放当前块内的图像描述符数据的指针数组
						const float *mat_I_point_array_CPU[image_count_per_block];
						//host存放当前块数据哈希计算结果的指针数组
						float *hash_base_array_CPU[image_count_per_block];
						//device存放当前块内的图像描述符数据的指针数组
						float *mat_I_point_array_GPU[image_count_per_block];
						//device存放当前块数据哈希计算结果的指针数组
						float *hash_base_array_GPU[image_count_per_block];
						//保存整个块内数据描述符的数量大小

						//host存放预读块内的图像描述符数据的指针数组
						const float *mat_I_pre_point_array_CPU[image_count_per_block];
						//host存放预读块内数据哈希计算结果的指针数组
						//float *hash_base_pre_array_CPU[image_count_per_block];
						//device存放预读块内的图像描述符数据的指针数组
						float *mat_I_pre_point_array_GPU[image_count_per_block];
						//device存放预读块数据哈希计算结果的指针数组
						//float *hash_base_pre_array_GPU[image_count_per_block];

						//initialize all pointers
						for (int i = 0; i < image_count_per_block; i++) {
							mat_I_point_array_CPU[i] = NULL;
							hash_base_array_CPU[i] = NULL;
							mat_I_point_array_GPU[i] = NULL;
							hash_base_array_GPU[i] = NULL;

							mat_I_pre_point_array_CPU[i] = NULL;
							//hash_base_pre_array_CPU[i] = NULL;
							mat_I_pre_point_array_GPU[i] = NULL;
							//hash_base_pre_array_GPU[i] = NULL;
						}

						for (int secondIter = 0; secondIter < block_count_per_group; secondIter++) {
							//处理每一块的数据，保证大概能把GPU内存塞满(或许应该多次试验看哪一组实验效果最好)
							{
								if (secondIter == 0) {
									//store all hash result for this block
									std::map<IndexT, HashedDescriptions> hashed_base_;
									//处理当前块内每一张图片的数据后上传到GPU上
									for (int m = 0; m < image_count_per_block; ++m)
									{
										std::set<IndexT>::const_iterator iter = used_index.begin();
										std::advance(iter, m + secondIter*image_count_per_block);
										const IndexT I = *iter;
										const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
										const unsigned char * tabI =
											reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
										const size_t dimension = regionsI->DescriptorLength();

										Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
										//descriptions minus zero_mean_descriptors before upload 
										Eigen::MatrixXf descriptionsMat;
										descriptionsMat = mat_I.template cast<float>();
										for (int k = 0; k < descriptionsMat.cols(); k++) {
											descriptionsMat.col(k) -= _zero_mean_descriptor;
										}
										float *descriptionsMat_data_temp = descriptionsMat.data();
										mat_I_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
										//upload mat_I_array_CPU[m]
										{
											int descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
											cudaMalloc((void **)&mat_I_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
											cudaMemcpy(mat_I_point_array_GPU[m], mat_I_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
										}
										mat_I_cols[m] = mat_I.cols();

									}
									//处理预读块内每一张图片的数据后上传到GPU上
									for (int m = 0; m < image_count_per_block; ++m)
									{
										std::set<IndexT>::const_iterator iter = used_index.begin();
										std::advance(iter, m + (secondIter + 1)*image_count_per_block);
										const IndexT I = *iter;
										const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
										const unsigned char * tabI =
											reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
										const size_t dimension = regionsI->DescriptorLength();

										Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
										//descriptions minus zero_mean_descriptors before upload 
										Eigen::MatrixXf descriptionsMat;
										descriptionsMat = mat_I.template cast<float>();
										for (int k = 0; k < descriptionsMat.cols(); k++) {
											descriptionsMat.col(k) -= _zero_mean_descriptor;
										}
										float *descriptionsMat_data_temp = descriptionsMat.data();
										mat_I_pre_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
										//upload mat_I_pre_array_CPU[m]
										{
											int descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
											cudaMalloc((void **)&mat_I_pre_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
											cudaMemcpy(mat_I_pre_point_array_GPU[m], mat_I_pre_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
										}
										mat_I_pre_cols[m] = mat_I.cols();
									}

									//为当前块数据做hash
									{
										for (int m = 0; m < image_count_per_block; ++m) {
											size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
											cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
											hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
												primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
											hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
											cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


											//free
											{
												cudaFree(hash_base_array_GPU[m]);
												mat_I_point_array_CPU[m] = NULL;
												hash_base_array_GPU[m] = NULL;
											}

											//Determine the bucket index for each IMG in a block.
											float *secondary_projection_CPU = NULL;
											float *secondary_projection_GPU = NULL;
											int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
											secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
											cudaMalloc((void **)secondary_projection_GPU,
												sizeof(float) * secondary_projection_CPU_size);
											//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
											secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
												//左式
												secondary_hash_projection_data_GPU[firstIter],
												//右式
												mat_I_point_array_GPU[m],
												//左式A行数
												myCascadeHasher.nb_bits_per_bucket_,
												//左式A列数
												myCascadeHasher.nb_hash_code_,
												//右式B列数
												mat_I_cols[m]
											);
											//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
											//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
											cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
												sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
											//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


											//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
											//(secondary_projection_CPU, secondary_projection_CPU_size);
											//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
											//(secondary_projection_CPU, secondary_projection_CPU_size);

											//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

											{
												int imgCountBeforeBlockM = 0;
												for (int sss = 0; sss < m; sss++) {
													imgCountBeforeBlockM += mat_I_cols[m];
												}
												for (int i = 0; i < mat_I_cols[m]; ++i) {
													// Allocate space for each bucket id.
													IndexT m_index = m;
													hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
													hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
													// Compute hash code.
													auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
													hash_code = stl::dynamic_bitset(descriptionDimension);
													for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
													{
														hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
													}

													// Determine the bucket index for each group.
													//Eigen::VectorXf secondary_projection;
													for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
													{
														uint16_t bucket_id = 0;



														for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
														{
															//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
															bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((imgCountBeforeBlockM + i)*(myCascadeHasher.nb_hash_code_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
														}
														hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
													}
												}

												//free
												{
													cudaFree(mat_I_point_array_GPU[m]);
													mat_I_point_array_GPU[m] = NULL;
													free(hash_base_array_CPU[m]);
													hash_base_array_CPU[m] = NULL;
												}

												// Build the Buckets
												{
													hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
													for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
													{
														hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

														// Add the descriptor ID to the proper bucket group and id.
														for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
														{
															const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
															hashed_base_[m].buckets[i][bucket_id].push_back(j);
														}
													}
												}
											}
											free(secondary_projection_CPU);
											secondary_projection_CPU = NULL;
										}
									}
									//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
									{
										char file_io_temp_i[2] = { ' ','\0' };
										file_io_temp_i[0] = secondIter + 48;
										const std::string file_io_str_i = file_io_temp_i;

										char file_name_temp[2] = { ' ','\0' };
										file_name_temp[0] = secondIter + 48;
										const std::string file_name_temp_m = file_name_temp;
										const std::string file_name_temp2 = "block_" + file_name_temp_m;
										const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
										if (!stlplus::file_exists(sHash)) {
											hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
										}
										else {
											std::cout << sHash << " already exists" << std::endl;
										}
									}
									//变换当前块与预读块的相关数据
									for (int m = 0; m < image_count_per_block; ++m) {
										mat_I_point_array_CPU[m] = mat_I_pre_point_array_CPU[m];
										mat_I_pre_point_array_CPU[m] = NULL;
										mat_I_cols[m] = mat_I_pre_cols[m];
										cudaMalloc((void**)&mat_I_point_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										cudaMalloc((void**)&hash_base_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										mat_I_point_array_GPU[m] = mat_I_pre_point_array_GPU[m];
									}


									//再预读一块数据进来
									//处理预读块内每一张图片的数据后上传到GPU上
									for (int m = 0; m < image_count_per_block; ++m)
									{
										std::set<IndexT>::const_iterator iter = used_index.begin();
										std::advance(iter, m + (secondIter + 2)*image_count_per_block);
										const IndexT I = *iter;
										const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
										const unsigned char * tabI =
											reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
										const size_t dimension = regionsI->DescriptorLength();

										Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
										//descriptions minus zero_mean_descriptors before upload 
										Eigen::MatrixXf descriptionsMat;
										descriptionsMat = mat_I.template cast<float>();
										for (int k = 0; k < descriptionsMat.cols(); k++) {
											descriptionsMat.col(k) -= _zero_mean_descriptor;
										}
										float *descriptionsMat_data_temp = descriptionsMat.data();
										mat_I_pre_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
										//upload mat_I_pre_array_CPU[m]
										{
											int descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
											cudaMalloc((void **)&mat_I_pre_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
											cudaMemcpy(mat_I_pre_point_array_GPU[m], mat_I_pre_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
										}
										mat_I_pre_cols[m] = mat_I.cols();
									}
								}
								else if (secondIter > 0 && secondIter < block_count_per_group - 2) {
									//store all hash result for this block
									std::map<IndexT, HashedDescriptions> hashed_base_;
									// 同步函数
									cudaThreadSynchronize();
									//为当前块数据做hash
									{
										for (int m = 0; m < image_count_per_block; ++m) {
											size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
											cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
											hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
												primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
											hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
											cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


											//free
											{
												cudaFree(hash_base_array_GPU[m]);
												mat_I_point_array_CPU[m] = NULL;
												hash_base_array_GPU[m] = NULL;
											}

											//Determine the bucket index for each IMG in a block.
											float *secondary_projection_CPU = NULL;
											float *secondary_projection_GPU = NULL;
											int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
											secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
											cudaMalloc((void **)secondary_projection_GPU,
												sizeof(float) * secondary_projection_CPU_size);
											//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
											secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
												//左式
												secondary_hash_projection_data_GPU[firstIter],
												//右式
												mat_I_point_array_GPU[m],
												//左式A行数
												myCascadeHasher.nb_bits_per_bucket_,
												//左式A列数
												myCascadeHasher.nb_hash_code_,
												//右式B列数
												mat_I_cols[m]
											);
											//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
											//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
											cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
												sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
											//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


											//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
											//(secondary_projection_CPU, secondary_projection_CPU_size);
											//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
											//(secondary_projection_CPU, secondary_projection_CPU_size);

											//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

											{
												int imgCountBeforeBlockM = 0;
												for (int sss = 0; sss < m; sss++) {
													imgCountBeforeBlockM += mat_I_cols[m];
												}
												for (int i = 0; i < mat_I_cols[m]; ++i) {
													// Allocate space for each bucket id.
													IndexT m_index = m;
													hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
													hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
													// Compute hash code.
													auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
													hash_code = stl::dynamic_bitset(descriptionDimension);
													for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
													{
														hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
													}

													// Determine the bucket index for each group.
													//Eigen::VectorXf secondary_projection;
													for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
													{
														uint16_t bucket_id = 0;



														for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
														{
															//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
															bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((imgCountBeforeBlockM + i)*(myCascadeHasher.nb_hash_code_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
														}
														hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
													}
												}

												//free
												{
													cudaFree(mat_I_point_array_GPU[m]);
													mat_I_point_array_GPU[m] = NULL;
													free(hash_base_array_CPU[m]);
													hash_base_array_CPU[m] = NULL;
												}

												// Build the Buckets
												{
													hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
													for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
													{
														hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

														// Add the descriptor ID to the proper bucket group and id.
														for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
														{
															const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
															hashed_base_[m].buckets[i][bucket_id].push_back(j);
														}
													}
												}
											}
											free(secondary_projection_CPU);
											secondary_projection_CPU = NULL;
										}
									}
									//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
									{
										char file_io_temp_i[2] = { ' ','\0' };
										file_io_temp_i[0] = secondIter + 48;
										const std::string file_io_str_i = file_io_temp_i;

										char file_name_temp[2] = { ' ','\0' };
										file_name_temp[0] = secondIter + 48;
										const std::string file_name_temp_m = file_name_temp;
										const std::string file_name_temp2 = "block_" + file_name_temp_m;
										const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
										if (!stlplus::file_exists(sHash)) {
											hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
										}
										else {
											std::cout << sHash << " already exists" << std::endl;
										}
									}
									//变换当前块与预读块的相关数据
									for (int m = 0; m < image_count_per_block; ++m) {
										mat_I_point_array_CPU[m] = mat_I_pre_point_array_CPU[m];
										mat_I_pre_point_array_CPU[m] = NULL;
										mat_I_cols[m] = mat_I_pre_cols[m];
										cudaMalloc((void**)&mat_I_point_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										cudaMalloc((void**)&hash_base_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										mat_I_point_array_GPU[m] = mat_I_pre_point_array_GPU[m];
									}
									//再预读一块数据进来
									//处理预读块内每一张图片的数据后上传到GPU上
									for (int m = 0; m < image_count_per_block; ++m)
									{
										std::set<IndexT>::const_iterator iter = used_index.begin();
										std::advance(iter, m + (secondIter + 2)*image_count_per_block);
										const IndexT I = *iter;
										const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
										const unsigned char * tabI =
											reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
										const size_t dimension = regionsI->DescriptorLength();

										Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
										//descriptions minus zero_mean_descriptors before upload 
										Eigen::MatrixXf descriptionsMat;
										descriptionsMat = mat_I.template cast<float>();
										for (int k = 0; k < descriptionsMat.cols(); k++) {
											descriptionsMat.col(k) -= _zero_mean_descriptor;
										}
										float *descriptionsMat_data_temp = descriptionsMat.data();
										mat_I_pre_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
										//upload mat_I_pre_array_CPU[m]
										{
											int descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
											cudaMalloc((void **)&mat_I_pre_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
											cudaMemcpy(mat_I_pre_point_array_GPU[m], mat_I_pre_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
										}
										mat_I_pre_cols[m] = mat_I.cols();
									}
								}
								else if (secondIter == block_count_per_group - 2) {
									//store all hash result for this block
									std::map<IndexT, HashedDescriptions> hashed_base_;
									// 同步函数
									cudaThreadSynchronize();
									//为当前块数据做hash
									{
										for (int m = 0; m < image_count_per_block; ++m) {
											size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
											cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
											hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
												primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
											hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
											cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


											//free
											{
												cudaFree(hash_base_array_GPU[m]);
												mat_I_point_array_CPU[m] = NULL;
												hash_base_array_GPU[m] = NULL;
											}

											//Determine the bucket index for each IMG in a block.
											float *secondary_projection_CPU = NULL;
											float *secondary_projection_GPU = NULL;
											int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
											secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
											cudaMalloc((void **)secondary_projection_GPU,
												sizeof(float) * secondary_projection_CPU_size);
											//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
											secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
												//左式
												secondary_hash_projection_data_GPU[firstIter],
												//右式
												mat_I_point_array_GPU[m],
												//左式A行数
												myCascadeHasher.nb_bits_per_bucket_,
												//左式A列数
												myCascadeHasher.nb_hash_code_,
												//右式B列数
												mat_I_cols[m]
											);
											//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
											//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
											cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
												sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
											//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


											//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
											//(secondary_projection_CPU, secondary_projection_CPU_size);
											//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
											//(secondary_projection_CPU, secondary_projection_CPU_size);

											//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

											{
												int imgCountBeforeBlockM = 0;
												for (int sss = 0; sss < m; sss++) {
													imgCountBeforeBlockM += mat_I_cols[m];
												}
												for (int i = 0; i < mat_I_cols[m]; ++i) {
													// Allocate space for each bucket id.
													IndexT m_index = m;
													hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
													hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
													// Compute hash code.
													auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
													hash_code = stl::dynamic_bitset(descriptionDimension);
													for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
													{
														hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
													}

													// Determine the bucket index for each group.
													//Eigen::VectorXf secondary_projection;
													for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
													{
														uint16_t bucket_id = 0;



														for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
														{
															//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((imgCountBeforeBlockM + i)*(myCascadeHasher.nb_hash_code_) + k)] > 0 ? 1 : 0);
															bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
														}
														hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
													}
												}

												//free
												{
													cudaFree(mat_I_point_array_GPU[m]);
													mat_I_point_array_GPU[m] = NULL;
													free(hash_base_array_CPU[m]);
													hash_base_array_CPU[m] = NULL;
												}

												// Build the Buckets
												{
													hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
													for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
													{
														hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

														// Add the descriptor ID to the proper bucket group and id.
														for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
														{
															const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
															hashed_base_[m].buckets[i][bucket_id].push_back(j);
														}
													}
												}
											}
											free(secondary_projection_CPU);
											secondary_projection_CPU = NULL;
										}
									}
									//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
									{
										char file_io_temp_i[2] = { ' ','\0' };
										file_io_temp_i[0] = secondIter + 48;
										const std::string file_io_str_i = file_io_temp_i;

										char file_name_temp[2] = { ' ','\0' };
										file_name_temp[0] = secondIter + 48;
										const std::string file_name_temp_m = file_name_temp;
										const std::string file_name_temp2 = "block_" + file_name_temp_m;
										const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
										if (!stlplus::file_exists(sHash)) {
											hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
										}
										else {
											std::cout << sHash << " already exists" << std::endl;
										}
									}
									//变换当前块与预读块的相关数据
									secondIter++;
									for (int m = 0; m < image_count_per_block; ++m) {
										mat_I_point_array_CPU[m] = mat_I_pre_point_array_CPU[m];
										mat_I_pre_point_array_CPU[m] = NULL;
										mat_I_cols[m] = mat_I_pre_cols[m];
										cudaMalloc((void**)&mat_I_point_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										cudaMalloc((void**)&hash_base_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										mat_I_point_array_GPU[m] = mat_I_pre_point_array_GPU[m];
									}
									//处理最后一块数据
									// 同步函数
									cudaThreadSynchronize();
									//为最后一块(当前的预读块)数据做hash
									{
										for (int m = 0; m < image_count_per_block; ++m) {
											size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
											cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
											hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
												primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
											hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
											cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


											//free
											{
												cudaFree(hash_base_array_GPU[m]);
												mat_I_point_array_CPU[m] = NULL;
												hash_base_array_GPU[m] = NULL;
											}

											//Determine the bucket index for each IMG in a block.
											float *secondary_projection_CPU = NULL;
											float *secondary_projection_GPU = NULL;
											int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
											secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
											cudaMalloc((void **)secondary_projection_GPU,
												sizeof(float) * secondary_projection_CPU_size);
											//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
											secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
												//左式
												secondary_hash_projection_data_GPU[firstIter],
												//右式
												mat_I_point_array_GPU[m],
												//左式A行数
												myCascadeHasher.nb_bits_per_bucket_,
												//左式A列数
												myCascadeHasher.nb_hash_code_,
												//右式B列数
												mat_I_cols[m]
											);
											//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
											//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
											cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
												sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
											//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


											//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
											//(secondary_projection_CPU, secondary_projection_CPU_size);
											//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
											//(secondary_projection_CPU, secondary_projection_CPU_size);

											//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

											{
												int imgCountBeforeBlockM = 0;
												for (int sss = 0; sss < m; sss++) {
													imgCountBeforeBlockM += mat_I_cols[m];
												}
												for (int i = 0; i < mat_I_cols[m]; ++i) {
													// Allocate space for each bucket id.
													IndexT m_index = m;
													hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
													hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
													// Compute hash code.
													auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
													hash_code = stl::dynamic_bitset(descriptionDimension);
													for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
													{
														hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
													}

													// Determine the bucket index for each group.
													//Eigen::VectorXf secondary_projection;
													for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
													{
														uint16_t bucket_id = 0;



														for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
														{
															//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((imgCountBeforeBlockM + i)*(myCascadeHasher.nb_hash_code_) + k)] > 0 ? 1 : 0);
															bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
														}
														hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
													}
												}

												//free
												{
													cudaFree(mat_I_point_array_GPU[m]);
													mat_I_point_array_GPU[m] = NULL;
													free(hash_base_array_CPU[m]);
													hash_base_array_CPU[m] = NULL;
												}

												// Build the Buckets
												{
													hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
													for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
													{
														hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

														// Add the descriptor ID to the proper bucket group and id.
														for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
														{
															const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
															hashed_base_[m].buckets[i][bucket_id].push_back(j);
														}
													}
												}
											}
											free(secondary_projection_CPU);
											secondary_projection_CPU = NULL;
										}
									}
									//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
									{
										char file_io_temp_i[2] = { ' ','\0' };
										file_io_temp_i[0] = secondIter + 48;
										const std::string file_io_str_i = file_io_temp_i;

										char file_name_temp[2] = { ' ','\0' };
										file_name_temp[0] = secondIter + 48;
										const std::string file_name_temp_m = file_name_temp;
										const std::string file_name_temp2 = "block_" + file_name_temp_m;
										const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
										if (!stlplus::file_exists(sHash)) {
											hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
										}
										else {
											std::cout << sHash << " already exists" << std::endl;
										}
									}
								}
								else {
									std::cout << "computing hash for group" << firstIter << "has done" << std::endl;
									return EXIT_FAILURE;
								}
							}
						}
					}
					std::cout << "Task (Regions Hashing for group " << firstIter << ") done in (s): " << timer.elapsed() << std::endl;
					//std::cout << "Task (Regions Hashing) done in (s): " << timer.elapsed() << std::endl;
				}
			}
			///////
			//处理完当前组数据后，再执行下面几行代码，把第二组的数据赋值给第一组
			sSfM_Data_Filename_hash = sSfM_Data_Filename_hash_pre;
			sMatchesOutputDir_hash = sMatchesOutputDir_hash_pre;
			sfm_data_hash = sfm_data_hash_pre;
			firstIter++;
			//---------------------------------------
			// 直接处理当前组的数据
			//---------------------------------------
			{
				//---------------------------------------
				// Load SfM Scene regions
				//---------------------------------------
				// Init the regions_type from the image describer file (used for image regions extraction)
				using namespace openMVG::features;
				const std::string sImage_describer = stlplus::create_filespec(sMatchesOutputDir_hash, "image_describer", "json");
				//The default regions_type is SIFT_Regions
				//The default SIFT_Regions is Scalar type
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

				if (!regions_provider->load(sfm_data_hash, sMatchesOutputDir_hash, regions_type, &progress)) {
					std::cerr << std::endl << "Invalid regions." << std::endl;
					return EXIT_FAILURE;
				}

				PairWiseMatches map_PutativesMatches;

				// Build some alias from SfM_Data Views data:
				// - List views as a vector of filenames & image sizes
				std::vector<std::string> vec_fileNames;
				std::vector<std::pair<size_t, size_t>> vec_imagesSize;
				{
					vec_fileNames.reserve(sfm_data_hash.GetViews().size());
					vec_imagesSize.reserve(sfm_data_hash.GetViews().size());
					for (Views::const_iterator iter = sfm_data_hash.GetViews().begin();
						iter != sfm_data_hash.GetViews().end();
						++iter)
					{
						const View * v = iter->second.get();
						vec_fileNames.push_back(stlplus::create_filespec(sfm_data_hash.s_root_path,
							v->s_Img_path));
						vec_imagesSize.push_back(std::make_pair(v->ui_width, v->ui_height));
					}
				}

				std::cout << std::endl << " - COMPUTE HASH(for 1 group) - " << std::endl;
				// If the matches already exists, reload them
				if (!bForce
					&& (stlplus::file_exists(sMatchesOutputDir_hash + "/matches.putative.txt")
						|| stlplus::file_exists(sMatchesOutputDir_hash + "/matches.putative.bin"))
					)
				{
					if (!(Load(map_PutativesMatches, sMatchesOutputDir_hash + "/matches.putative.bin") ||
						Load(map_PutativesMatches, sMatchesOutputDir_hash + "/matches.putative.txt")))
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
							//default set runs here
							std::cout << "Using FAST_CASCADE_HASHING_L2 matcher" << std::endl;
							//collectionMatcher.reset(new Cascade_Hashing_Matcher_Regions_GPU(fDistRatio));
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
						//std::cerr << "Invalid Nearest Neighbor method: " << sNearestMatchingMethod << std::endl;
						//return EXIT_FAILURE;
					}
					// Perform the cascade hashing
					system::Timer timer;
					{
						// Collect used view indexes for an image group
						std::set<IndexT> used_index;
						for (int i = 0; i < image_count_per_group; i++) {
							used_index.insert(i);
							//used_index.insert(firstIter * image_count_per_group + i);
						}
						//openMVG::matching_image_collection::Cascade_Hash_Generate sCascade_Hash_Generate;
						//第二层数据调度策略 CPU内存 <--> GPU内存
						//1.启用零拷贝内存
						cudaSetDeviceFlags(cudaDeviceMapHost);

						std::vector <int> mat_I_cols;
						mat_I_cols.resize(image_count_per_block);

						std::vector <int> mat_I_pre_cols;
						mat_I_pre_cols.resize(image_count_per_block);

						//host存放当前块内的图像描述符数据的指针数组
						const float *mat_I_point_array_CPU[image_count_per_block];
						//host存放当前块数据哈希计算结果的指针数组
						float *hash_base_array_CPU[image_count_per_block];
						//device存放当前块内的图像描述符数据的指针数组
						float *mat_I_point_array_GPU[image_count_per_block];
						//device存放当前块数据哈希计算结果的指针数组
						float *hash_base_array_GPU[image_count_per_block];
						//保存整个块内数据描述符的数量大小

						//host存放预读块内的图像描述符数据的指针数组
						const float *mat_I_pre_point_array_CPU[image_count_per_block];
						//host存放预读块内数据哈希计算结果的指针数组
						//float *hash_base_pre_array_CPU[image_count_per_block];
						//device存放预读块内的图像描述符数据的指针数组
						float *mat_I_pre_point_array_GPU[image_count_per_block];
						//device存放预读块数据哈希计算结果的指针数组
						//float *hash_base_pre_array_GPU[image_count_per_block];

						//initialize all pointers
						for (int i = 0; i < image_count_per_block; i++) {
							mat_I_point_array_CPU[i] = NULL;
							hash_base_array_CPU[i] = NULL;
							mat_I_point_array_GPU[i] = NULL;
							hash_base_array_GPU[i] = NULL;

							mat_I_pre_point_array_CPU[i] = NULL;
							//hash_base_pre_array_CPU[i] = NULL;
							mat_I_pre_point_array_GPU[i] = NULL;
							//hash_base_pre_array_GPU[i] = NULL;
						}

						for (int secondIter = 0; secondIter < block_count_per_group; secondIter++) {
							//处理每一块的数据，保证大概能把GPU内存塞满(或许应该多次试验看哪一组实验效果最好)
							{
								if (secondIter == 0) {
									//store all hash result for this block
									std::map<IndexT, HashedDescriptions> hashed_base_;
									//处理当前块内每一张图片的数据后上传到GPU上
									for (int m = 0; m < image_count_per_block; ++m)
									{
										std::set<IndexT>::const_iterator iter = used_index.begin();
										std::advance(iter, m + secondIter*image_count_per_block);
										const IndexT I = *iter;
										const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
										const unsigned char * tabI =
											reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
										const size_t dimension = regionsI->DescriptorLength();

										Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
										//descriptions minus zero_mean_descriptors before upload 
										Eigen::MatrixXf descriptionsMat;
										descriptionsMat = mat_I.template cast<float>();
										for (int k = 0; k < descriptionsMat.cols(); k++) {
											descriptionsMat.col(k) -= _zero_mean_descriptor;
										}
										float *descriptionsMat_data_temp = descriptionsMat.data();
										mat_I_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
										//upload mat_I_array_CPU[m]
										{
											int descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
											cudaMalloc((void **)&mat_I_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
											cudaMemcpy(mat_I_point_array_GPU[m], mat_I_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
										}
										mat_I_cols[m] = mat_I.cols();

									}
									//处理预读块内每一张图片的数据后上传到GPU上
									for (int m = 0; m < image_count_per_block; ++m)
									{
										std::set<IndexT>::const_iterator iter = used_index.begin();
										std::advance(iter, m + (secondIter + 1)*image_count_per_block);
										const IndexT I = *iter;
										const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
										const unsigned char * tabI =
											reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
										const size_t dimension = regionsI->DescriptorLength();

										Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
										//descriptions minus zero_mean_descriptors before upload 
										Eigen::MatrixXf descriptionsMat;
										descriptionsMat = mat_I.template cast<float>();
										for (int k = 0; k < descriptionsMat.cols(); k++) {
											descriptionsMat.col(k) -= _zero_mean_descriptor;
										}
										float *descriptionsMat_data_temp = descriptionsMat.data();
										mat_I_pre_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
										//upload mat_I_pre_array_CPU[m]
										{
											int descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
											cudaMalloc((void **)&mat_I_pre_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
											cudaMemcpy(mat_I_pre_point_array_GPU[m], mat_I_pre_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
										}
										mat_I_pre_cols[m] = mat_I.cols();
									}

									//为当前块数据做hash
									{
										for (int m = 0; m < image_count_per_block; ++m) {
											size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
											cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
											hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
												primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
											hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
											cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


											//free
											{
												cudaFree(hash_base_array_GPU[m]);
												mat_I_point_array_CPU[m] = NULL;
												hash_base_array_GPU[m] = NULL;
											}

											//Determine the bucket index for each IMG in a block.
											float *secondary_projection_CPU = NULL;
											float *secondary_projection_GPU = NULL;
											int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
											secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
											cudaMalloc((void **)secondary_projection_GPU,
												sizeof(float) * secondary_projection_CPU_size);
											//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
											secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
												//左式
												secondary_hash_projection_data_GPU[firstIter],
												//右式
												mat_I_point_array_GPU[m],
												//左式A行数
												myCascadeHasher.nb_bits_per_bucket_,
												//左式A列数
												myCascadeHasher.nb_hash_code_,
												//右式B列数
												mat_I_cols[m]
											);
											//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
											//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
											cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
												sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
											//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


											//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
											//(secondary_projection_CPU, secondary_projection_CPU_size);
											//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
											//(secondary_projection_CPU, secondary_projection_CPU_size);

											//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

											{
												int imgCountBeforeBlockM = 0;
												for (int sss = 0; sss < m; sss++) {
													imgCountBeforeBlockM += mat_I_cols[m];
												}
												for (int i = 0; i < mat_I_cols[m]; ++i) {
													// Allocate space for each bucket id.
													IndexT m_index = m;
													hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
													hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
													// Compute hash code.
													auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
													hash_code = stl::dynamic_bitset(descriptionDimension);
													for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
													{
														hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
													}

													// Determine the bucket index for each group.
													//Eigen::VectorXf secondary_projection;
													for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
													{
														uint16_t bucket_id = 0;



														for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
														{
															//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
															bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((imgCountBeforeBlockM + i)*(myCascadeHasher.nb_hash_code_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
														}
														hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
													}
												}

												//free
												{
													cudaFree(mat_I_point_array_GPU[m]);
													mat_I_point_array_GPU[m] = NULL;
													free(hash_base_array_CPU[m]);
													hash_base_array_CPU[m] = NULL;
												}

												// Build the Buckets
												{
													hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
													for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
													{
														hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

														// Add the descriptor ID to the proper bucket group and id.
														for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
														{
															const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
															hashed_base_[m].buckets[i][bucket_id].push_back(j);
														}
													}
												}
											}
											free(secondary_projection_CPU);
											secondary_projection_CPU = NULL;
										}
									}
									//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
									{
										char file_io_temp_i[2] = { ' ','\0' };
										file_io_temp_i[0] = secondIter + 48;
										const std::string file_io_str_i = file_io_temp_i;

										char file_name_temp[2] = { ' ','\0' };
										file_name_temp[0] = secondIter + 48;
										const std::string file_name_temp_m = file_name_temp;
										const std::string file_name_temp2 = "block_" + file_name_temp_m;
										const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
										if (!stlplus::file_exists(sHash)) {
											hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
										}
										else {
											std::cout << sHash << " already exists" << std::endl;
										}
									}
									//变换当前块与预读块的相关数据
									for (int m = 0; m < image_count_per_block; ++m) {
										mat_I_point_array_CPU[m] = mat_I_pre_point_array_CPU[m];
										mat_I_pre_point_array_CPU[m] = NULL;
										mat_I_cols[m] = mat_I_pre_cols[m];
										cudaMalloc((void**)&mat_I_point_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										cudaMalloc((void**)&hash_base_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										mat_I_point_array_GPU[m] = mat_I_pre_point_array_GPU[m];
									}


									//再预读一块数据进来
									//处理预读块内每一张图片的数据后上传到GPU上
									for (int m = 0; m < image_count_per_block; ++m)
									{
										std::set<IndexT>::const_iterator iter = used_index.begin();
										std::advance(iter, m + (secondIter + 2)*image_count_per_block);
										const IndexT I = *iter;
										const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
										const unsigned char * tabI =
											reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
										const size_t dimension = regionsI->DescriptorLength();

										Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
										//descriptions minus zero_mean_descriptors before upload 
										Eigen::MatrixXf descriptionsMat;
										descriptionsMat = mat_I.template cast<float>();
										for (int k = 0; k < descriptionsMat.cols(); k++) {
											descriptionsMat.col(k) -= _zero_mean_descriptor;
										}
										float *descriptionsMat_data_temp = descriptionsMat.data();
										mat_I_pre_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
										//upload mat_I_pre_array_CPU[m]
										{
											int descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
											cudaMalloc((void **)&mat_I_pre_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
											cudaMemcpy(mat_I_pre_point_array_GPU[m], mat_I_pre_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
										}
										mat_I_pre_cols[m] = mat_I.cols();
									}
								}
								else if (secondIter > 0 && secondIter < block_count_per_group - 2) {
									//store all hash result for this block
									std::map<IndexT, HashedDescriptions> hashed_base_;
									// 同步函数
									cudaThreadSynchronize();
									//为当前块数据做hash
									{
										for (int m = 0; m < image_count_per_block; ++m) {
											size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
											cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
											hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
												primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
											hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
											cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


											//free
											{
												cudaFree(hash_base_array_GPU[m]);
												mat_I_point_array_CPU[m] = NULL;
												hash_base_array_GPU[m] = NULL;
											}

											//Determine the bucket index for each IMG in a block.
											float *secondary_projection_CPU = NULL;
											float *secondary_projection_GPU = NULL;
											int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
											secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
											cudaMalloc((void **)secondary_projection_GPU,
												sizeof(float) * secondary_projection_CPU_size);
											//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
											secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
												//左式
												secondary_hash_projection_data_GPU[firstIter],
												//右式
												mat_I_point_array_GPU[m],
												//左式A行数
												myCascadeHasher.nb_bits_per_bucket_,
												//左式A列数
												myCascadeHasher.nb_hash_code_,
												//右式B列数
												mat_I_cols[m]
											);
											//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
											//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
											cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
												sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
											//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


											//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
											//(secondary_projection_CPU, secondary_projection_CPU_size);
											//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
											//(secondary_projection_CPU, secondary_projection_CPU_size);

											//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

											{
												int imgCountBeforeBlockM = 0;
												for (int sss = 0; sss < m; sss++) {
													imgCountBeforeBlockM += mat_I_cols[m];
												}
												for (int i = 0; i < mat_I_cols[m]; ++i) {
													// Allocate space for each bucket id.
													IndexT m_index = m;
													hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
													hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
													// Compute hash code.
													auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
													hash_code = stl::dynamic_bitset(descriptionDimension);
													for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
													{
														hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
													}

													// Determine the bucket index for each group.
													//Eigen::VectorXf secondary_projection;
													for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
													{
														uint16_t bucket_id = 0;



														for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
														{
															//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
															bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((imgCountBeforeBlockM + i)*(myCascadeHasher.nb_hash_code_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
														}
														hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
													}
												}

												//free
												{
													cudaFree(mat_I_point_array_GPU[m]);
													mat_I_point_array_GPU[m] = NULL;
													free(hash_base_array_CPU[m]);
													hash_base_array_CPU[m] = NULL;
												}

												// Build the Buckets
												{
													hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
													for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
													{
														hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

														// Add the descriptor ID to the proper bucket group and id.
														for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
														{
															const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
															hashed_base_[m].buckets[i][bucket_id].push_back(j);
														}
													}
												}
											}
											free(secondary_projection_CPU);
											secondary_projection_CPU = NULL;
										}
									}
									//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
									{
										char file_io_temp_i[2] = { ' ','\0' };
										file_io_temp_i[0] = secondIter + 48;
										const std::string file_io_str_i = file_io_temp_i;

										char file_name_temp[2] = { ' ','\0' };
										file_name_temp[0] = secondIter + 48;
										const std::string file_name_temp_m = file_name_temp;
										const std::string file_name_temp2 = "block_" + file_name_temp_m;
										const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
										if (!stlplus::file_exists(sHash)) {
											hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
										}
										else {
											std::cout << sHash << " already exists" << std::endl;
										}
									}
									//变换当前块与预读块的相关数据
									for (int m = 0; m < image_count_per_block; ++m) {
										mat_I_point_array_CPU[m] = mat_I_pre_point_array_CPU[m];
										mat_I_pre_point_array_CPU[m] = NULL;
										mat_I_cols[m] = mat_I_pre_cols[m];
										cudaMalloc((void**)&mat_I_point_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										cudaMalloc((void**)&hash_base_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										mat_I_point_array_GPU[m] = mat_I_pre_point_array_GPU[m];
									}
									//再预读一块数据进来
									//处理预读块内每一张图片的数据后上传到GPU上
									for (int m = 0; m < image_count_per_block; ++m)
									{
										std::set<IndexT>::const_iterator iter = used_index.begin();
										std::advance(iter, m + (secondIter + 2)*image_count_per_block);
										const IndexT I = *iter;
										const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
										const unsigned char * tabI =
											reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
										const size_t dimension = regionsI->DescriptorLength();

										Eigen::Map<BaseMat> mat_I((unsigned char*)tabI, dimension, regionsI->RegionCount());
										//descriptions minus zero_mean_descriptors before upload 
										Eigen::MatrixXf descriptionsMat;
										descriptionsMat = mat_I.template cast<float>();
										for (int k = 0; k < descriptionsMat.cols(); k++) {
											descriptionsMat.col(k) -= _zero_mean_descriptor;
										}
										float *descriptionsMat_data_temp = descriptionsMat.data();
										mat_I_pre_point_array_CPU[m] = reinterpret_cast<const float*>(descriptionsMat_data_temp);
										//upload mat_I_pre_array_CPU[m]
										{
											int descriptionsMatSize = descriptionsMat.rows() *descriptionsMat.cols();
											cudaMalloc((void **)&mat_I_pre_point_array_GPU[m], sizeof(float)*descriptionsMatSize);
											cudaMemcpy(mat_I_pre_point_array_GPU[m], mat_I_pre_point_array_CPU[m], sizeof(float)*descriptionsMatSize, cudaMemcpyHostToDevice);
										}
										mat_I_pre_cols[m] = mat_I.cols();
									}
								}
								else if (secondIter == block_count_per_group - 2) {
									//store all hash result for this block
									std::map<IndexT, HashedDescriptions> hashed_base_;
									// 同步函数
									cudaThreadSynchronize();
									//为当前块数据做hash
									{
										for (int m = 0; m < image_count_per_block; ++m) {
											size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
											cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
											hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
												primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
											hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
											cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


											//free
											{
												cudaFree(hash_base_array_GPU[m]);
												mat_I_point_array_CPU[m] = NULL;
												hash_base_array_GPU[m] = NULL;
											}

											//Determine the bucket index for each IMG in a block.
											float *secondary_projection_CPU = NULL;
											float *secondary_projection_GPU = NULL;
											int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
											secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
											cudaMalloc((void **)secondary_projection_GPU,
												sizeof(float) * secondary_projection_CPU_size);
											//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
											secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
												//左式
												secondary_hash_projection_data_GPU[firstIter],
												//右式
												mat_I_point_array_GPU[m],
												//左式A行数
												myCascadeHasher.nb_bits_per_bucket_,
												//左式A列数
												myCascadeHasher.nb_hash_code_,
												//右式B列数
												mat_I_cols[m]
											);
											//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
											//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
											cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
												sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
											//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


											//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
											//(secondary_projection_CPU, secondary_projection_CPU_size);
											//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
											//(secondary_projection_CPU, secondary_projection_CPU_size);

											//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

											{
												int imgCountBeforeBlockM = 0;
												for (int sss = 0; sss < m; sss++) {
													imgCountBeforeBlockM += mat_I_cols[m];
												}
												for (int i = 0; i < mat_I_cols[m]; ++i) {
													// Allocate space for each bucket id.
													IndexT m_index = m;
													hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
													hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
													// Compute hash code.
													auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
													hash_code = stl::dynamic_bitset(descriptionDimension);
													for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
													{
														hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
													}

													// Determine the bucket index for each group.
													//Eigen::VectorXf secondary_projection;
													for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
													{
														uint16_t bucket_id = 0;



														for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
														{
															//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
															bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((imgCountBeforeBlockM + i)*(myCascadeHasher.nb_hash_code_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
														}
														hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
													}
												}

												//free
												{
													cudaFree(mat_I_point_array_GPU[m]);
													mat_I_point_array_GPU[m] = NULL;
													free(hash_base_array_CPU[m]);
													hash_base_array_CPU[m] = NULL;
												}

												// Build the Buckets
												{
													hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
													for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
													{
														hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

														// Add the descriptor ID to the proper bucket group and id.
														for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
														{
															const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
															hashed_base_[m].buckets[i][bucket_id].push_back(j);
														}
													}
												}
											}
											free(secondary_projection_CPU);
											secondary_projection_CPU = NULL;
										}
									}
									//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
									{
										char file_io_temp_i[2] = { ' ','\0' };
										file_io_temp_i[0] = secondIter + 48;
										const std::string file_io_str_i = file_io_temp_i;

										char file_name_temp[2] = { ' ','\0' };
										file_name_temp[0] = secondIter + 48;
										const std::string file_name_temp_m = file_name_temp;
										const std::string file_name_temp2 = "block_" + file_name_temp_m;
										const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
										if (!stlplus::file_exists(sHash)) {
											hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
										}
										else {
											std::cout << sHash << " already exists" << std::endl;
										}
									}
									//变换当前块与预读块的相关数据
									for (int m = 0; m < image_count_per_block; ++m) {
										mat_I_point_array_CPU[m] = mat_I_pre_point_array_CPU[m];
										mat_I_pre_point_array_CPU[m] = NULL;
										mat_I_cols[m] = mat_I_pre_cols[m];
										cudaMalloc((void**)&mat_I_point_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										cudaMalloc((void**)&hash_base_array_GPU[m], sizeof(float) * mat_I_cols[m] * descriptionDimension);
										mat_I_point_array_GPU[m] = mat_I_pre_point_array_GPU[m];
									}
									//处理最后一块数据
									// 同步函数
									cudaThreadSynchronize();
									secondIter++;
									//为最后一块(当前的预读块)数据做hash
									{
										for (int m = 0; m < image_count_per_block; ++m) {
											size_t hash_base_array_GPU_size = mat_I_cols[m] * descriptionDimension;
											cudaMalloc((void **)&hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size);
											hash_base_array_GPU[m] = myCascadeHasher.hash_gen(mat_I_cols[m], descriptionDimension,
												primary_hash_projection_data_device, mat_I_point_array_GPU[m]);
											hash_base_array_CPU[m] = (float *)malloc(hash_base_array_GPU_size * (sizeof(float)));
											cudaMemcpy(hash_base_array_CPU[m], hash_base_array_GPU[m], sizeof(float)*hash_base_array_GPU_size, cudaMemcpyDeviceToHost);


											//free
											{
												cudaFree(hash_base_array_GPU[m]);
												mat_I_point_array_CPU[m] = NULL;
												hash_base_array_GPU[m] = NULL;
											}

											//Determine the bucket index for each IMG in a block.
											float *secondary_projection_CPU = NULL;
											float *secondary_projection_GPU = NULL;
											int secondary_projection_CPU_size = myCascadeHasher.nb_bits_per_bucket_ * mat_I_cols[m];
											secondary_projection_CPU = (float *)malloc(sizeof(float) * secondary_projection_CPU_size);
											cudaMalloc((void **)secondary_projection_GPU,
												sizeof(float) * secondary_projection_CPU_size);
											//secondary_projection = myCascadeHasher.secondary_hash_projection_[j] * mat_I_point_array_CPU[m];
											secondary_projection_GPU = myCascadeHasher.determine_buket_index_for_each_group(
												//左式
												secondary_hash_projection_data_GPU[firstIter],
												//右式
												mat_I_point_array_GPU[m],
												//左式A行数
												myCascadeHasher.nb_bits_per_bucket_,
												//左式A列数
												myCascadeHasher.nb_hash_code_,
												//右式B列数
												mat_I_cols[m]
											);
											//得到的结果是secondary_projectiob * 一张图像上描述符数据得到的结果，用于决定哈希桶的数值
											//矩阵应该是一个行列数分别为myCascadeHasher.nb_bits_per_bucket_ 、 mat_I_cols[m]的float矩阵
											cudaMemcpy(secondary_projection_CPU, secondary_projection_GPU,
												sizeof(float)*secondary_projection_CPU_size, cudaMemcpyDeviceToHost);
											//Eigen::MatrixXf secondary_projection_mat = myCascadeHasher.secondary_hash_projection_[firstIter] * tempMat;


											//Eigen::MatrixXf secondary_projection = Eigen::Map<Eigen::Matrix<float, myCascadeHasher.nb_bits_per_bucket_ , mat_I_cols[m]>>(secondary_projection_CPU);
											//(secondary_projection_CPU, secondary_projection_CPU_size);
											//Eigen::VectorXf secondary_projection = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>
											//(secondary_projection_CPU, secondary_projection_CPU_size);

											//计算出真正的哈希结果存放到 std::map<IndexT, HashedDescriptions> hashed_base_ 里面

											{
												int imgCountBeforeBlockM = 0;
												for (int sss = 0; sss < m; sss++) {
													imgCountBeforeBlockM += mat_I_cols[m];
												}
												for (int i = 0; i < mat_I_cols[m]; ++i) {
													// Allocate space for each bucket id.
													IndexT m_index = m;
													hashed_base_[m_index].hashed_desc.resize(mat_I_cols[m]);
													hashed_base_[m_index].hashed_desc[i].bucket_ids.resize(myCascadeHasher.nb_bucket_groups_);
													// Compute hash code.
													auto& hash_code = hashed_base_[m].hashed_desc[i].hash_code;
													hash_code = stl::dynamic_bitset(descriptionDimension);
													for (int j = 0; j < myCascadeHasher.nb_hash_code_; ++j)
													{
														hash_code[j] = hash_base_array_CPU[m][(i*(myCascadeHasher.nb_hash_code_) + j)] > 0;
													}

													// Determine the bucket index for each group.
													//Eigen::VectorXf secondary_projection;
													for (int j = 0; j < myCascadeHasher.nb_bucket_groups_; ++j)
													{
														uint16_t bucket_id = 0;



														for (int k = 0; k < myCascadeHasher.nb_bits_per_bucket_; ++k)
														{
															//bucket_id = (bucket_id << 1) + (secondary_projection_mat(k, i) > 0 ? 1 : 0);
															bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((i)*(myCascadeHasher.nb_bits_per_bucket_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection_CPU[((imgCountBeforeBlockM + i)*(myCascadeHasher.nb_hash_code_) + k)] > 0 ? 1 : 0);
															//bucket_id = (bucket_id << 1) + (secondary_projection(k) > 0 ? 1 : 0);
														}
														hashed_base_[m_index].hashed_desc[i].bucket_ids[j] = bucket_id;
													}
												}

												//free
												{
													cudaFree(mat_I_point_array_GPU[m]);
													mat_I_point_array_GPU[m] = NULL;
													free(hash_base_array_CPU[m]);
													hash_base_array_CPU[m] = NULL;
												}

												// Build the Buckets
												{
													hashed_base_[m].buckets.resize(myCascadeHasher.nb_bucket_groups_);
													for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; ++i)
													{
														hashed_base_[m].buckets[i].resize(myCascadeHasher.nb_buckets_per_group_);

														// Add the descriptor ID to the proper bucket group and id.
														for (int j = 0; j < hashed_base_[m].hashed_desc.size(); ++j)
														{
															const uint16_t bucket_id = hashed_base_[m].hashed_desc[j].bucket_ids[i];
															hashed_base_[m].buckets[i][bucket_id].push_back(j);
														}
													}
												}
											}
											free(secondary_projection_CPU);
											secondary_projection_CPU = NULL;
										}
									}
									//将std::map<IndexT, HashedDescriptions> hashed_base_(也就是一整块的数据哈希处理的结果)存放到文件当中去
									{
										char file_io_temp_i[2] = { ' ','\0' };
										file_io_temp_i[0] = secondIter + 48;
										const std::string file_io_str_i = file_io_temp_i;

										char file_name_temp[2] = { ' ','\0' };
										file_name_temp[0] = secondIter + 48;
										const std::string file_name_temp_m = file_name_temp;
										const std::string file_name_temp2 = "block_" + file_name_temp_m;
										const std::string sHash = stlplus::create_filespec(sMatchesOutputDir_hash, stlplus::basename_part(file_name_temp2), "hash");
										if (!stlplus::file_exists(sHash)) {
											hashed_code_file_io::write_hashed_base(sHash, hashed_base_);
										}
										else {
											std::cout << sHash << " already exists" << std::endl;
										}
									}
								}
								else {
									std::cerr << "error when index the secondIter!:" << secondIter << std::endl;
									return EXIT_FAILURE;
								}
							}
						}
					}
					std::cout << "Task (Regions Hashing for group " << firstIter << ") done in (s): " << timer.elapsed() << std::endl;
				}
			}
			///////
		}
		else {
			std::cerr << "error when index the firstIter!:" << firstIter <<std::endl;
			return EXIT_FAILURE;
		}
		
	}
	//free
	{
		primary_hash_projection_data = NULL;
		cudaFree(primary_hash_projection_data_device);
		primary_hash_projection_data_device = NULL;
		
		for (int i = 0; i < myCascadeHasher.nb_bucket_groups_; i++) {
			secondary_hash_projection_data_CPU[i] = NULL;
			cudaFree(secondary_hash_projection_data_GPU[i]);
			secondary_hash_projection_data_GPU[i] = NULL;
		}
	}
	std::cout << "compute hash for all groups success!" << std::endl;
	return EXIT_SUCCESS;
}

Pair_Set getBetweenBlockPairs(int startImgIndexThisBlockL, int startImgIndexThisBlockR)
{
	Pair_Set pairs;
	for (IndexT I = startImgIndexThisBlockL; I < static_cast<IndexT>(startImgIndexThisBlockL + image_count_per_block); ++I)
	{
		for (IndexT J = startImgIndexThisBlockR; J < static_cast<IndexT>(startImgIndexThisBlockR + image_count_per_block); ++J)
		{
			pairs.insert({ I,J });
		}
	}
	return pairs;
}

Pair_Set getInsideBlockPairs(int startImgIndexThisBlock) 
{
	Pair_Set pairs;
	for (IndexT I = startImgIndexThisBlock; I < static_cast<IndexT>(startImgIndexThisBlock + image_count_per_block); ++I)
	{
		for (IndexT J = I + 1; J < static_cast<IndexT>(startImgIndexThisBlock + image_count_per_block); ++J) 
		{
			pairs.insert({I,J});
		}
	}
	return pairs;
}

//块内自我匹配
void match_block_itself
(
	PairWiseMatches &map_PutativesMatches,
	const sfm::Regions_Provider & regions_provider,
	std::string matches_final_result_dir,
	std::string filename_hash_mid_result,
	int secondIter,
	int startImgIndexThisBlock,
	std::map<openMVG::IndexT, HashedDescriptions> hashed_base_
)
{
	//当前组目录
	if (matches_final_result_dir.empty() || !stlplus::is_folder(matches_final_result_dir)) {
		std::cerr << "\nIt is an invalid output directory" << std::endl;
		return;
	}

	//当前 块哈希结果文件名
	if (!stlplus::file_exists(filename_hash_mid_result)) {
		std::cerr << "\nIt is an invalid input hashcode mid result file!" << std::endl;
		return;
	}
	/*std::map<openMVG::IndexT, HashedDescriptions> hashed_base_;
	hashed_code_file_io::read_hashed_base(filename_hash_mid_result, hashed_base_);*/
	
	Pair_Set pairs = getInsideBlockPairs(startImgIndexThisBlock);
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
	openMVG::matching::CascadeHasherGPU cascade_hasher;
	if (!used_index.empty())
	{
		const IndexT I = secondIter;
		const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
		const size_t dimension = regionsI->DescriptorLength();
		cascade_hasher.Init(dimension);
	}

	// Perform matching between all the pairs
	for (const auto & pairs : map_Pairs) 
	{
		int temp = (pairs.first) % image_count_per_group;
		const IndexT I = temp;
		const std::vector<IndexT> & indexToCompare = pairs.second;

		const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);

		const std::vector<features::PointFeature> pointFeaturesI = regionsI->GetRegionsPositions();
		const unsigned char * tabI =
			reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
		const size_t dimension = regionsI->DescriptorLength();
		Eigen::Map<BaseMat> mat_I((unsigned char *)tabI, dimension, regionsI->RegionCount());

		for (int j = 0; j < (int)indexToCompare.size(); ++j) 
		{
			int tempJ = (indexToCompare[j]) % image_count_per_group;
			const size_t J = tempJ;
			const std::shared_ptr<features::Regions> regionsJ = regions_provider.get(J);
			if (regionsJ == nullptr) 
			{
				std::cout <<"error when --regions_provider.get(J)--" << std::endl;
			}
			// Matrix representation of the query input data;
			const unsigned char * tabJ = reinterpret_cast<const unsigned char*>(regionsJ->DescriptorRawData());
			Eigen::Map<BaseMat> mat_J((unsigned char*)tabJ, dimension, regionsJ->RegionCount());

			IndMatches pvec_indices;
			using ResultType = typename Accumulator<unsigned char>::Type;
			std::vector<ResultType> pvec_distances;
			pvec_distances.reserve(regionsJ->RegionCount() * 2);
			pvec_indices.reserve(regionsJ->RegionCount() * 2);

			// Match the query descriptors to the database
			// ResultType = float
			// using BaseMat = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
			cascade_hasher.Match_HashedDescriptions<BaseMat, ResultType>(
				hashed_base_[J], mat_J,
				hashed_base_[I], mat_I,
				&pvec_indices, &pvec_distances);

			std::vector<int> vec_nn_ratio_idx;
			// Filter the matches using a distance ratio test:
			//   The probability that a match is correct is determined by taking
			//   the ratio of distance from the closest neighbor to the distance
			//   of the second closest.
			float fDistRatioGPU = 0.8f;
			matching::NNdistanceRatio(
				pvec_distances.begin(), // distance start
				pvec_distances.end(),   // distance end
				2, // Number of neighbor in iterator sequence (minimum required 2)
				vec_nn_ratio_idx, // output (indices that respect the distance Ratio)
				Square(fDistRatioGPU));

			

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

			//// Draw correspondences after Nearest Neighbor ratio filter
			//{
			//	const std::string jpg_filenameL = "E:\\/imageData/tianjin/DJI_0/DJI_0001.JPG";
			//	const std::string jpg_filenameR = "E:\\/imageData/tianjin/DJI_0/DJI_0002.JPG";
			//	Image<unsigned char> imageL, imageR;
			//	ReadImage(jpg_filenameL.c_str(), &imageL);
			//	ReadImage(jpg_filenameR.c_str(), &imageR);
			//	assert(imageL.data() && imageR.data());

			//	std::unique_ptr<features::Regions> sss;
			//	const bool bVertical = true;
			//	Matches2SVG
			//	(
			//		jpg_filenameL,
			//		{ imageL.Width(), imageL.Height() },
			//		regions_provider.get(0)->GetRegionsPositions(),
			//		jpg_filenameR,
			//		{ imageR.Width(), imageR.Height() },
			//		regions_provider.get(1)->GetRegionsPositions(),
			//		vec_putative_matches,
			//		"03_Matches.svg",
			//		bVertical
			//	);
			//}

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
			//++(*my_progress_bar);
		}
	}
	
}
//同组不同块之间互相匹配(没有块内的图片之间互相匹配)
void matchBetweenBlocksInOneGroup
(
	PairWiseMatches &map_PutativesMatches,//存储匹配结果
	const sfm::Regions_Provider & regions_provider,//本组图像的特征描述符指针
	std::map<openMVG::IndexT, HashedDescriptions> hashed_base_,//每块hash结果数据
	std::map<openMVG::IndexT, HashedDescriptions> hashed_base_next,//待匹配的块hash结果数据
	int secondIter,//第一块编号 
	int secondIterNext,//第二块编号
	int startImgIndexThisBlock,//第一块内的起始图片编号
	int startImgIndexThisBlockNext//待匹配块内的起始图片编号
)
{
	Pair_Set pairs = getBetweenBlockPairs(startImgIndexThisBlock, startImgIndexThisBlockNext);
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
	openMVG::matching::CascadeHasherGPU cascade_hasher;
	if (!used_index.empty())
	{
		const IndexT I = secondIter;
		const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
		const size_t dimension = regionsI->DescriptorLength();
		cascade_hasher.Init(dimension);
	}

	// Perform matching between all the pairs
	for (const auto & pairs : map_Pairs)
	{
		int tempF = (pairs.first) % image_count_per_group;
		const IndexT I = tempF;

		std::vector<IndexT> tempS;
		tempS.resize(pairs.second.size());
		for (int i = 0; i < pairs.second.size(); i++) {
			int temp = (pairs.second[i]) % image_count_per_group;
			tempS[i] = temp;
		}
		const std::vector<IndexT> & indexToCompare = tempS;

		const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);

		const std::vector<features::PointFeature> pointFeaturesI = regionsI->GetRegionsPositions();
		const unsigned char * tabI =
			reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
		const size_t dimension = regionsI->DescriptorLength();
		Eigen::Map<BaseMat> mat_I((unsigned char *)tabI, dimension, regionsI->RegionCount());

		for (int j = 0; j < (int)indexToCompare.size(); ++j)
		{
			int temp_Index = indexToCompare[j] % image_count_per_group;
			const size_t J = temp_Index;
			const std::shared_ptr<features::Regions> regionsJ = regions_provider.get(J);

			// Matrix representation of the query input data;
			const unsigned char * tabJ = reinterpret_cast<const unsigned char*>(regionsJ->DescriptorRawData());
			Eigen::Map<BaseMat> mat_J((unsigned char*)tabJ, dimension, regionsJ->RegionCount());

			IndMatches pvec_indices;
			using ResultType = typename Accumulator<unsigned char>::Type;
			std::vector<ResultType> pvec_distances;
			pvec_distances.reserve(regionsJ->RegionCount() * 2);
			pvec_indices.reserve(regionsJ->RegionCount() * 2);

			// Match the query descriptors to the database
			// ResultType = float
			// using BaseMat = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
			
			cascade_hasher.Match_HashedDescriptions<BaseMat, ResultType>(
				hashed_base_next[J], mat_J,
				hashed_base_[I], mat_I,
				&pvec_indices, &pvec_distances);

			std::vector<int> vec_nn_ratio_idx;
			// Filter the matches using a distance ratio test:
			//   The probability that a match is correct is determined by taking
			//   the ratio of distance from the closest neighbor to the distance
			//   of the second closest.
			float fDistRatioGPU = 0.8f;
			matching::NNdistanceRatio(
				pvec_distances.begin(), // distance start
				pvec_distances.end(),   // distance end
				2, // Number of neighbor in iterator sequence (minimum required 2)
				vec_nn_ratio_idx, // output (indices that respect the distance Ratio)
				Square(fDistRatioGPU));

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
			//++(*my_progress_bar);
		}
	}
}
//不同组不同块之间互相匹配(没有块内的图片之间互相匹配)
void matchBetweenBlocksInDiffGroups
(
	PairWiseMatches &map_PutativesMatches,//存储匹配结果
	const sfm::Regions_Provider & regions_provider,//第一组图像的特征描述符指针
	const sfm::Regions_Provider & regions_provider_next,//待匹配组图像的特征描述符指针
	std::map<openMVG::IndexT, HashedDescriptions> hashed_base_,//每块hash结果数据
	std::map<openMVG::IndexT, HashedDescriptions> hashed_base_next,//待匹配的块hash结果数据
	int secondIter,//第一块编号 
	int secondIterNext,//第二块编号
	int startImgIndexThisBlock,//第一块内的起始图片编号
	int startImgIndexThisBlockNext//待匹配块内的起始图片编号
)
{
	Pair_Set pairs = getBetweenBlockPairs(startImgIndexThisBlock, startImgIndexThisBlockNext);
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
	openMVG::matching::CascadeHasherGPU cascade_hasher;
	if (!used_index.empty())
	{
		const IndexT I = secondIter;
		const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);
		const size_t dimension = regionsI->DescriptorLength();
		cascade_hasher.Init(dimension);
	}

	// Perform matching between all the pairs
	for (const auto & pairs : map_Pairs)
	{
		int tempF = (pairs.first) % image_count_per_group;
		const IndexT I = tempF;

		std::vector<IndexT> tempS;
		tempS.resize(pairs.second.size());
		for (int i = 0; i < pairs.second.size(); i++) {
			int temp = (pairs.second[i]) % image_count_per_group;
			tempS[i] = temp;
		}
		const std::vector<IndexT> & indexToCompare = tempS;

		const std::shared_ptr<features::Regions> regionsI = regions_provider.get(I);

		const std::vector<features::PointFeature> pointFeaturesI = regionsI->GetRegionsPositions();
		const unsigned char * tabI =
			reinterpret_cast<const unsigned char*>(regionsI->DescriptorRawData());
		const size_t dimension = regionsI->DescriptorLength();
		Eigen::Map<BaseMat> mat_I((unsigned char *)tabI, dimension, regionsI->RegionCount());

		for (int j = 0; j < (int)indexToCompare.size(); ++j)
		{
			int temp_Index = indexToCompare[j] % image_count_per_group;
			const size_t J = temp_Index;
			const std::shared_ptr<features::Regions> regionsJ = regions_provider_next.get(J);

			// Matrix representation of the query input data;
			const unsigned char * tabJ = reinterpret_cast<const unsigned char*>(regionsJ->DescriptorRawData());
			Eigen::Map<BaseMat> mat_J((unsigned char*)tabJ, dimension, regionsJ->RegionCount());

			IndMatches pvec_indices;
			using ResultType = typename Accumulator<unsigned char>::Type;
			std::vector<ResultType> pvec_distances;
			pvec_distances.reserve(regionsJ->RegionCount() * 2);
			pvec_indices.reserve(regionsJ->RegionCount() * 2);

			// Match the query descriptors to the database
			// ResultType = float
			// using BaseMat = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
			/*std::cout << "I: " << I << std::endl;
			std::cout << "J: " << J << std::endl;*/
			cascade_hasher.Match_HashedDescriptions<BaseMat, ResultType>(
				hashed_base_next[J], mat_J,
				hashed_base_[I], mat_I,
				&pvec_indices, &pvec_distances);

			std::vector<int> vec_nn_ratio_idx;
			// Filter the matches using a distance ratio test:
			//   The probability that a match is correct is determined by taking
			//   the ratio of distance from the closest neighbor to the distance
			//   of the second closest.
			float fDistRatioGPU = 0.8f;
			matching::NNdistanceRatio(
				pvec_distances.begin(), // distance start
				pvec_distances.end(),   // distance end
				2, // Number of neighbor in iterator sequence (minimum required 2)
				vec_nn_ratio_idx, // output (indices that respect the distance Ratio)
				Square(fDistRatioGPU));

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
			//++(*my_progress_bar);
		}
	}
}
//组内匹配
//1.组内的块之间互相匹配
//2.组内的块内自我匹配
void matchForThisGroup
(
	PairWiseMatches &map_PutativesMatches,//存储一整组自我匹配的匹配结果
	int firstIter,//第一组的编号
	std::string matches_final_result_dir,//第一组特征描述符文件的目录
	const sfm::Regions_Provider & regions_provider//第一组图像的特征描述符指针
)
{
	// If the matches already exists, reload them
	if ((stlplus::file_exists(matches_final_result_dir + "/matches.putative_itself.txt")
		|| stlplus::file_exists(matches_final_result_dir + "/matches.putative_itself.bin"))
		)
	{
		if (!(Load(map_PutativesMatches, matches_final_result_dir + "/matches.putative_itself.bin") ||
			Load(map_PutativesMatches, matches_final_result_dir + "/matches.putative_itself.txt")))
		{
			std::cerr << "Cannot load input matches file";
			return;
		}
		std::cout << "\t PREVIOUS RESULTS LOADED;"
			<< " #pair: " << map_PutativesMatches.size() << std::endl;
	}
	else 
	{
		////读取一整组的特征描述符数据
		if (matches_final_result_dir.empty() || !stlplus::is_folder(matches_final_result_dir))
		{
			std::cerr << "\nIt is an invalid output directory" << std::endl;
			return;
		}
		//---------------------------------------
		// Read SfM Scene (image view & intrinsics data)
		//---------------------------------------
		std::string sfm_data_filename = matches_final_result_dir + "sfm_data.json";
		SfM_Data sfm_data;
		if (!Load(sfm_data, sfm_data_filename, ESfM_Data(VIEWS | INTRINSICS))) {
			std::cerr << std::endl
				<< "The input SfM_Data file \"" << sfm_data_filename << "\" cannot be read." << std::endl;
			return;
		}
		//---------------------------------------
		// Load SfM Scene regions
		//---------------------------------------
		// Init the regions_type from the image describer file (used for image regions extraction)
		using namespace openMVG::features;
		const std::string sImage_describer = stlplus::create_filespec(matches_final_result_dir, "image_describer", "json");
		std::unique_ptr<Regions> regions_type = Init_region_type_from_file(sImage_describer);
		if (!regions_type)
		{
			std::cerr << "Invalid: "
				<< sImage_describer << " regions type file." << std::endl;
			return;
		}
		// Load the corresponding view regions
		std::shared_ptr<Regions_Provider> regions_provider;
		// Default regions provider (load & store all regions in memory)
		regions_provider = std::make_shared<Regions_Provider>();
		// Show the progress on the command line:
		C_Progress_display progress;

		if (!regions_provider->load(sfm_data, matches_final_result_dir, regions_type, &progress)) {
			std::cerr << std::endl << "Invalid regions." << std::endl;
			return;
		}

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
		std::cout << std::endl << " - PUTATIVE MATCHES FOR GROUP" << firstIter << " ITSELF- " << std::endl;
		//块之间进行匹配
		{
			for (int secondIter = 0; secondIter < block_count_per_group; secondIter++)
			{
				for (int secondIterNext = secondIter + 1; secondIterNext < block_count_per_group; secondIterNext++)
				{
					std::cout << std::endl << " - PUTATIVE MATCHES BETWEEN BLOCKS INSIDE THE SAME GROUP- " << std::endl;
					//处理好第一块哈希 结果文件名
					char temp_j[2] = { ' ','\0' };
					temp_j[0] = secondIter + 48;
					const std::string str_j = temp_j;
					std::string filename_hash_mid_result = matches_final_result_dir + "block_" + str_j + ".hash";

					std::map<openMVG::IndexT, HashedDescriptions> hashed_base_;
					hashed_code_file_io::read_hashed_base(filename_hash_mid_result, hashed_base_);

					int startImgIndexThisBlock = 0;
					startImgIndexThisBlock = firstIter * image_count_per_group + secondIter*image_count_per_block;

					//处理好待匹配块哈希 结果文件名
					char temp_j_next[2] = { ' ','\0' };
					temp_j_next[0] = secondIterNext + 48;
					const std::string str_j_next = temp_j_next;
					std::string filename_hash_mid_result_next = matches_final_result_dir + "block_" + str_j_next + ".hash";

					std::map<openMVG::IndexT, HashedDescriptions> hashed_base_next;
					hashed_code_file_io::read_hashed_base(filename_hash_mid_result_next, hashed_base_next);

					int startImgIndexThisBlockNext = 0;
					startImgIndexThisBlockNext = firstIter * image_count_per_group + secondIterNext*image_count_per_block;
					matchBetweenBlocksInOneGroup(
						map_PutativesMatches, //匹配结果
						(*regions_provider.get()), //当前组特征描述符数据
						hashed_base_,//每块hash结果数据
						hashed_base_next,//待匹配的hash结果数据
						secondIter,//第一块编号
						secondIterNext,//第二块编号
						startImgIndexThisBlock,//第一块内的起始图片编号
						startImgIndexThisBlockNext//待匹配块内的起始图片编号
					);
				}
			}
		}
		//每个块自我匹配
		{
			std::cout << std::endl << " - PUTATIVE MATCHES INSIDE BLOCKS - " << std::endl;
			// Perform the matching
			system::Timer timer;
			{
				for (int secondIter = 0; secondIter < block_count_per_group; secondIter++)
				{
					//处理好 块哈希 结果文件名
					char temp_j[2] = { ' ','\0' };
					temp_j[0] = secondIter + 48;
					const std::string str_j = temp_j;
					std::string filename_hash_mid_result = matches_final_result_dir + "block_" + str_j + ".hash";

					std::map<openMVG::IndexT, HashedDescriptions> hashed_base_;
					hashed_code_file_io::read_hashed_base(filename_hash_mid_result, hashed_base_);

					int startImgIndexThisBlock = 0;
					startImgIndexThisBlock = firstIter * image_count_per_group + secondIter*image_count_per_block;
					match_block_itself(map_PutativesMatches, *regions_provider.get(), matches_final_result_dir, filename_hash_mid_result, secondIter, startImgIndexThisBlock, hashed_base_);
				}
			}
			std::cout << "Task (Regions Matching for group " << firstIter << ") done in (s): " << timer.elapsed() << std::endl;
		}

		//先过滤？
	}
}
//组间匹配
//1.与其他组内的块互相匹配
//2.组内自我匹配()
void matchBetweenGroups
(
	int firstIter,
	int firstIterNext,
	std::string matches_final_result_dir,
	std::string matches_final_result_dir_next,
	std::shared_ptr<Regions_Provider> regions_provider,
	std::shared_ptr<Regions_Provider> regions_provider_next
)
{
	//开始匹配
	std::cout << std::endl << " - PUTATIVE MATCHES FOR GROUP " << firstIter << " BETWEEN GROUP " << firstIterNext << std::endl;

	//存储一整组与待匹配组的匹配结果文件名称
	std::string matches_result_filaname;

	char temp_group[2] = { ' ','\0' };
	temp_group[0] = firstIter + 48;
	const std::string str_group = temp_group;

	char temp_group_next[2] = { ' ','\0' };
	temp_group_next[0] = firstIterNext + 48;
	const std::string str_group_next = temp_group_next;

	matches_result_filaname = matches_final_result_dir + "/matches.putative.group_" + str_group + "_between_group_" + str_group_next + ".bin";
	//存储一整组与待匹配组匹配的匹配结果
	PairWiseMatches map_PutativesMatches;

	//当前块、待匹配块和预读块的哈希中间结果存储文件名
	std::string filename_hash_mid_result;
	std::string filename_hash_mid_result_next;
	std::string filename_hash_mid_result_pre;

	//当前块、待匹配块和预读块的哈希中间结果
	std::map<openMVG::IndexT, HashedDescriptions> hashed_base_;
	std::map<openMVG::IndexT, HashedDescriptions> hashed_base_next;
	std::map<openMVG::IndexT, HashedDescriptions> hashed_base_pre;

	int startIndexThisGroup = firstIter * image_count_per_group;
	int startIndexThisGroupNext = firstIterNext * image_count_per_group;

	// If the matches already exists, reload them
	if (stlplus::file_exists(matches_result_filaname))
	{
		if (!(Load(map_PutativesMatches, matches_result_filaname)))
		{
			std::cerr << "Cannot load input matches file";
			return;
		}
		std::cout << "\t PREVIOUS RESULTS LOADED;"
			<< " #pair: " << map_PutativesMatches.size() << std::endl;
	}
	else
	{
		if (matches_final_result_dir.empty() || !stlplus::is_folder(matches_final_result_dir))
		{
			std::cerr << "\nIt is an invalid output directory" << std::endl;
			return;
		}
		if (matches_final_result_dir_next.empty() || !stlplus::is_folder(matches_final_result_dir_next))
		{
			std::cerr << "\nIt(next) is an invalid output directory" << std::endl;
			return;
		}

		//两组之间的块执行匹配
		for (int thisGroupBlockIndex = 0; thisGroupBlockIndex < block_count_per_group; thisGroupBlockIndex++)
		{
			//读当前块(thisGroupBlockIndex)的哈希结果文件
			char temp[2] = { ' ','\0' };
			temp[0] = thisGroupBlockIndex + 48;
			const std::string str = temp;
			filename_hash_mid_result = matches_final_result_dir + "/block_" + str + ".hash";

			hashed_code_file_io::read_hashed_base(filename_hash_mid_result, hashed_base_);

			for (int nextGroupBlockIndex = 0; nextGroupBlockIndex < block_count_per_group; nextGroupBlockIndex++)
			{
				if (nextGroupBlockIndex == 0)
				{
					//读待匹配块(nextGroupBlockIndex)的哈希结果文件
					char temp_next[2] = { ' ','\0' };
					temp_next[0] = nextGroupBlockIndex + 48;
					const std::string str_next = temp_next;
					filename_hash_mid_result_next = matches_final_result_dir_next + "/block_" + str_next + ".hash";

					hashed_code_file_io::read_hashed_base(filename_hash_mid_result_next, hashed_base_next);
					//预读一个块(nextGroupBlockIndex+1)的哈希结果文件进来
					char temp_pre[2] = { ' ','\0' };
					temp_pre[0] = nextGroupBlockIndex + 1 + 48;
					std::string str_pre = temp_pre;
					filename_hash_mid_result_pre = matches_final_result_dir_next + "/block_" + str_pre + ".hash";

					hashed_code_file_io::read_hashed_base(filename_hash_mid_result_pre, hashed_base_pre);
					//匹配两个块的数据
					matchBetweenBlocksInDiffGroups
					(
						map_PutativesMatches,//存储匹配结果
						*regions_provider.get(),//第一组图像的特征描述符指针
						*regions_provider_next.get(),//待匹配组图像的特征描述符指针
						hashed_base_, //第一块哈希结果
						hashed_base_next, //待匹配块的哈希结果
						thisGroupBlockIndex,//第一块编号 
						nextGroupBlockIndex,//第二块编号
						startIndexThisGroup + thisGroupBlockIndex*image_count_per_block,//第一块内的起始图片编号
						startIndexThisGroupNext + nextGroupBlockIndex*image_count_per_block//待匹配块内的起始图片编号
					);
					//交换待匹配块(nextGroupBlockIndex)哈希数据和预读块(nextGroupBlockIndex+1)哈希数据
					hashed_base_next = hashed_base_pre;
					//记得更新startIndexThisGroupNext
					//再预读进来一块(nextGroupBlockIndex+2)新的哈希数据
					temp_pre[0] = nextGroupBlockIndex + 2 + 48;
					str_pre = temp_pre;
					filename_hash_mid_result_pre = matches_final_result_dir_next + "/block_" + str_pre + ".hash";

					hashed_code_file_io::read_hashed_base(filename_hash_mid_result_pre, hashed_base_pre);
				}
				else if (nextGroupBlockIndex < block_count_per_group - 2)
				{
					//匹配当前块和预读块的数据
					matchBetweenBlocksInDiffGroups
					(
						map_PutativesMatches,//存储匹配结果
						*regions_provider.get(),//第一组图像的特征描述符指针
						*regions_provider_next.get(),//待匹配组图像的特征描述符指针
						hashed_base_, //第一块哈希结果
						hashed_base_next, //待匹配块的哈希结果
						thisGroupBlockIndex,//第一块编号 
						nextGroupBlockIndex,//第二块编号
						startIndexThisGroup + thisGroupBlockIndex*image_count_per_block,//第一块内的起始图片编号
						startIndexThisGroupNext + nextGroupBlockIndex*image_count_per_block//待匹配块内的起始图片编号
					);
					//交换待匹配块(nextGroupBlockIndex)哈希数据和预读块(nextGroupBlockIndex+1)哈希数据
					hashed_base_next = hashed_base_pre;
					//记得更新startIndexThisGroupNext
					//再预读进来一块(nextGroupBlockIndex+2)新的哈希数据
					char temp_pre[2] = { ' ','\0' };
					temp_pre[0] = nextGroupBlockIndex + 2 + 48;
					std::string str_pre = temp_pre;

					filename_hash_mid_result_pre = matches_final_result_dir_next + "/block_" + str_pre + ".hash";

					hashed_code_file_io::read_hashed_base(filename_hash_mid_result_pre, hashed_base_pre);
				}
				else if (nextGroupBlockIndex == block_count_per_group - 2)
				{
					//匹配当前块和预读块的数据
					matchBetweenBlocksInDiffGroups
					(
						map_PutativesMatches,//存储匹配结果
						*regions_provider.get(),//第一组图像的特征描述符指针
						*regions_provider_next.get(),//待匹配组图像的特征描述符指针
						hashed_base_, //第一块哈希结果
						hashed_base_next, //待匹配块的哈希结果
						thisGroupBlockIndex,//第一块编号 
						nextGroupBlockIndex,//第二块编号
						startIndexThisGroup + thisGroupBlockIndex*image_count_per_block,//第一块内的起始图片编号
						startIndexThisGroupNext + nextGroupBlockIndex*image_count_per_block//待匹配块内的起始图片编号
					);
					//交换待匹配块(nextGroupBlockIndex)哈希数据和预读块(nextGroupBlockIndex+1)哈希数据
					hashed_base_next = hashed_base_pre;
					//记得更新一些数值
					nextGroupBlockIndex++;
					//第一个块和最后一个块执行匹配
					matchBetweenBlocksInDiffGroups
					(
						map_PutativesMatches,//存储匹配结果
						*regions_provider.get(),//第一组图像的特征描述符指针
						*regions_provider_next.get(),//待匹配组图像的特征描述符指针
						hashed_base_, //第一块哈希结果
						hashed_base_next, //待匹配块的哈希结果
						thisGroupBlockIndex,//第一块编号 
						nextGroupBlockIndex,//第二块编号
						startIndexThisGroup + thisGroupBlockIndex*image_count_per_block,//第一块内的起始图片编号
						startIndexThisGroupNext + nextGroupBlockIndex*image_count_per_block//待匹配块内的起始图片编号
					);
				}
				else
				{
					std::cout <<"match for block " << thisGroupBlockIndex <<"between block " << nextGroupBlockIndex << std::endl;
				}
			}
		}

		//---------------------------------------
		//-- Export putative matches
		//---------------------------------------
		if (!Save(map_PutativesMatches, std::string(matches_result_filaname)))
		{
			std::cerr
				<< "Cannot save computed matches in: "
				<< std::string(matches_final_result_dir + "/matches.putative_itself.bin");
			return;
		}
	}
}
	

//1.group_count组之间匹配
//2.组内的block_count_per_group块 自我组内匹配

int computeMatches::computeMatches() {
	

	//fundamental matrix
	std::string sGeometricModel = "f";
	//lowe's filter radio
	float fDistRatio = 0.8f;
	int iMatchingVideoMode = -1;
	std::string sPredefinedPairList = "";
	std::string sNearestMatchingMethod = "AUTO";
	bool bForce = false;
	bool bGuided_matching = false;
	int imax_iteration = 2048;
	unsigned int ui_max_cache_size = 0;

	//当前组目录
	std::string filename_hash_mid_result;
	std::string matches_final_result_dir;
	//待匹配组目录
	std::string filename_hash_mid_result_next;
	std::string matches_final_result_dir_next;
	//预读组目录
	std::string filename_hash_mid_result_pre;
	std::string matches_final_result_dir_pre;

	std::cout << " You called : " << "\n"
		<< "computeMatches" << "\n"
		/*<< "--input_file " << sSfM_Data_Filename_hash << "\n"
		<< "--out_dir " << sMatchesOutputDir_hash << "\n"*/
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
	// none of use
	if (sPredefinedPairList.length()) {
		ePairmode = PAIR_FROM_FILE;
		if (iMatchingVideoMode>0) {
			std::cerr << "\nIncompatible options: --videoModeMatching and --pairList" << std::endl;
			return EXIT_FAILURE;
		}
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
		default:
			std::cerr << "Unknown geometric model" << std::endl;
			return EXIT_FAILURE;
	}

	//在组的尺度上执行匹配
	//这里只做第一层数据调度
	for (int firstIter = 0; firstIter < group_count - 2; firstIter++) 
	{
		//read descriptions read descriptions read descriptions read descriptions read descriptions read descriptions
		char temp_firstIter[2] = { ' ','\0' };
		temp_firstIter[0] = firstIter + 48;
		const std::string str_firstIter = temp_firstIter;

		matches_final_result_dir = sSfM_Data_FilenameDir_father + "DJI_" + str_firstIter + "_build/";
		if (matches_final_result_dir.empty() || !stlplus::is_folder(matches_final_result_dir))
		{
			std::cerr << "\nIt is an invalid output directory" << std::endl;
			return EXIT_FAILURE;
		}
		//---------------------------------------
		// Read SfM Scene (image view & intrinsics data)
		//---------------------------------------
		std::string sfm_data_filename = matches_final_result_dir + "sfm_data.json";
		SfM_Data sfm_data;
		if (!Load(sfm_data, sfm_data_filename, ESfM_Data(VIEWS | INTRINSICS))) {
			std::cerr << std::endl
				<< "The input SfM_Data file \"" << sfm_data_filename << "\" cannot be read." << std::endl;
			return EXIT_FAILURE;
		}
		//---------------------------------------
		// Load SfM Scene regions
		//---------------------------------------
		// Init the regions_type from the image describer file (used for image regions extraction)
		using namespace openMVG::features;
		const std::string sImage_describer = stlplus::create_filespec(matches_final_result_dir, "image_describer", "json");
		std::unique_ptr<Regions> regions_type = Init_region_type_from_file(sImage_describer);
		if (!regions_type)
		{
			std::cerr << "Invalid: "
				<< sImage_describer << " regions type file." << std::endl;
			return EXIT_FAILURE;
		}
		// Load the corresponding view regions
		std::shared_ptr<Regions_Provider> regions_provider;
		// Default regions provider (load & store all regions in memory)
		regions_provider = std::make_shared<Regions_Provider>();
		// Show the progress on the command line:
		C_Progress_display progress;

		if (!regions_provider->load(sfm_data, matches_final_result_dir, regions_type, &progress)) {
			std::cerr << std::endl << "Invalid regions." << std::endl;
			return EXIT_FAILURE;
		}

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
		//read descriptions read descriptions read descriptions read descriptions read descriptions read descriptions

		//待匹配块数据
		std::string sfm_data_filename_next;
		SfM_Data sfm_data_next;
		// Load the corresponding view regions
		std::shared_ptr<Regions_Provider> regions_provider_next;
		// Show the progress on the command line:
		C_Progress_display progress_next;
		std::vector<std::string> vec_fileNames_next;
		std::vector<std::pair<size_t, size_t>> vec_imagesSize_next;

		//预读块数据
		std::string sfm_data_filename_pre;
		SfM_Data sfm_data_pre;
		// Load the corresponding view regions
		std::shared_ptr<Regions_Provider> regions_provider_pre;
		// Show the progress on the command line:
		C_Progress_display progress_pre;
		std::vector<std::string> vec_fileNames_pre;
		std::vector<std::pair<size_t, size_t>> vec_imagesSize_pre;
		if (firstIter < group_count - 3) 
		{
			for (int firstIterNext = firstIter + 1; firstIterNext < group_count; firstIterNext++)
			{
				if (firstIterNext == firstIter + 1)
				{
					//读待匹配(firstIterNext)的一组特征描述符
					//read descriptions read descriptions read descriptions read descriptions read descriptions read descriptions
					char temp_firstIterNext[2] = { ' ','\0' };
					temp_firstIterNext[0] = firstIterNext + 48;
					const std::string str_firstIterNext = temp_firstIterNext;

					matches_final_result_dir_next = sSfM_Data_FilenameDir_father + "DJI_" + str_firstIterNext + "_build/";
					if (matches_final_result_dir_next.empty() || !stlplus::is_folder(matches_final_result_dir_next))
					{
						std::cerr << "\nIt is an invalid output directory" << std::endl;
						return EXIT_FAILURE;
					}
					//---------------------------------------
					// Read SfM Scene (image view & intrinsics data)
					//---------------------------------------
					sfm_data_filename_next = matches_final_result_dir_next + "sfm_data.json";
					if (!Load(sfm_data_next, sfm_data_filename_next, ESfM_Data(VIEWS | INTRINSICS))) {
						std::cerr << std::endl
							<< "The input SfM_Data file \"" << sfm_data_filename << "\" cannot be read." << std::endl;
						return EXIT_FAILURE;
					}
					//---------------------------------------
					// Load SfM Scene regions
					//---------------------------------------
					// Init the regions_type from the image describer file (used for image regions extraction)
					using namespace openMVG::features;
					// Default regions provider (load & store all regions in memory)
					regions_provider_next = std::make_shared<Regions_Provider>();

					if (!regions_provider_next->load(sfm_data_next, matches_final_result_dir_next, regions_type, &progress_next)) {
						std::cerr << std::endl << "Invalid regions." << std::endl;
						return EXIT_FAILURE;
					}

					// Build some alias from SfM_Data Views data:
					// - List views as a vector of filenames & image sizes
					{
						vec_fileNames_next.reserve(sfm_data_next.GetViews().size());
						vec_imagesSize_next.reserve(sfm_data_next.GetViews().size());
						for (Views::const_iterator iter = sfm_data_next.GetViews().begin();
							iter != sfm_data_next.GetViews().end();
							++iter)
						{
							const View * v = iter->second.get();
							vec_fileNames_next.push_back(stlplus::create_filespec(sfm_data_next.s_root_path,
								v->s_Img_path));
							vec_imagesSize_next.push_back(std::make_pair(v->ui_width, v->ui_height));
						}
					}
					//read descriptions read descriptions read descriptions read descriptions read descriptions read descriptions

					//读预读组(firstIterNext+1)的特征描述符数据
					//read descriptions read descriptions read descriptions read descriptions read descriptions read descriptions
					char temp_firstIterPre[2] = { ' ','\0' };
					temp_firstIterPre[0] = firstIterNext + 1 + 48;
					std::string str_firstIterPre = temp_firstIterPre;

					matches_final_result_dir_pre = sSfM_Data_FilenameDir_father + "DJI_" + str_firstIterPre + "_build/";
					if (matches_final_result_dir_pre.empty() || !stlplus::is_folder(matches_final_result_dir_pre))
					{
						std::cerr << "\nIt is an invalid output directory" << std::endl;
						return EXIT_FAILURE;
					}
					//---------------------------------------
					// Read SfM Scene (image view & intrinsics data)
					//---------------------------------------
					sfm_data_filename_pre = matches_final_result_dir_pre + "sfm_data.json";
					if (!Load(sfm_data_pre, sfm_data_filename_pre, ESfM_Data(VIEWS | INTRINSICS))) {
						std::cerr << std::endl
							<< "The input SfM_Data file \"" << sfm_data_filename_pre << "\" cannot be read." << std::endl;
						return EXIT_FAILURE;
					}
					//---------------------------------------
					// Load SfM Scene regions
					//---------------------------------------
					// Init the regions_type from the image describer file (used for image regions extraction)
					using namespace openMVG::features;
					// Default regions provider (load & store all regions in memory)
					regions_provider_pre = std::make_shared<Regions_Provider>();

					if (!regions_provider_pre->load(sfm_data_pre, matches_final_result_dir_pre, regions_type, &progress_pre)) {
						std::cerr << "Invalid regions." << std::endl;
						return EXIT_FAILURE;
					}

					// Build some alias from SfM_Data Views data:
					// - List views as a vector of filenames & image sizes
					{
						vec_fileNames_pre.reserve(sfm_data_pre.GetViews().size());
						vec_imagesSize_pre.reserve(sfm_data_pre.GetViews().size());
						for (Views::const_iterator iter = sfm_data_pre.GetViews().begin();
							iter != sfm_data_pre.GetViews().end();
							++iter)
						{
							const View * v = iter->second.get();
							vec_fileNames_pre.push_back(stlplus::create_filespec(sfm_data_pre.s_root_path,
								v->s_Img_path));
							vec_imagesSize_pre.push_back(std::make_pair(v->ui_width, v->ui_height));
						}
					}
					//read descriptions read descriptions read descriptions read descriptions read descriptions read descriptions

					matchBetweenGroups
					(
						firstIter,
						firstIterNext,
						matches_final_result_dir,
						matches_final_result_dir_next,
						regions_provider,
						regions_provider_next
					);
					//预读组数据赋值给待匹配组数据，再预读一组(firstIterNext+2)特征描述符数据进来
					matches_final_result_dir_next = matches_final_result_dir_pre;
					regions_provider_next = regions_provider_pre;
					vec_fileNames_next = vec_fileNames_pre;
					vec_imagesSize_next = vec_imagesSize_pre;

					//read descriptions read descriptions read descriptions read descriptions read descriptions read descriptions
					//char temp_firstIterPre[2] = { ' ','\0' };
					temp_firstIterPre[0] = firstIterNext + 2 + 48;
					str_firstIterPre = temp_firstIterPre;

					matches_final_result_dir_pre = sSfM_Data_FilenameDir_father + "DJI_" + str_firstIterPre + "_build/";
					if (matches_final_result_dir_pre.empty() || !stlplus::is_folder(matches_final_result_dir_pre))
					{
						std::cerr << "\nIt is an invalid output directory" << std::endl;
						return EXIT_FAILURE;
					}
					//---------------------------------------
					// Read SfM Scene (image view & intrinsics data)
					//---------------------------------------
					sfm_data_filename_pre = matches_final_result_dir_pre + "sfm_data.json";
					//SfM_Data sfm_data_pre;
					if (!Load(sfm_data_pre, sfm_data_filename_pre, ESfM_Data(VIEWS | INTRINSICS))) {
						std::cerr << std::endl
							<< "The input SfM_Data file \"" << sfm_data_filename_pre << "\" cannot be read." << std::endl;
						return EXIT_FAILURE;
					}
					//---------------------------------------
					// Load SfM Scene regions
					//---------------------------------------
					// Init the regions_type from the image describer file (used for image regions extraction)
					using namespace openMVG::features;
					// Load the corresponding view regions
					//std::shared_ptr<Regions_Provider> regions_provider_pre;
					// Default regions provider (load & store all regions in memory)
					regions_provider_pre = std::make_shared<Regions_Provider>();
					// Show the progress on the command line:
					//C_Progress_display progress_pre;

					if (!regions_provider_pre->load(sfm_data_pre, matches_final_result_dir_pre, regions_type, &progress_pre)) {
						std::cerr << "Invalid regions." << std::endl;
						return EXIT_FAILURE;
					}

					// Build some alias from SfM_Data Views data:
					// - List views as a vector of filenames & image sizes
					/*std::vector<std::string> vec_fileNames_pre;
					std::vector<std::pair<size_t, size_t>> vec_imagesSize_pre;*/
					{
						vec_fileNames_pre.reserve(sfm_data_pre.GetViews().size());
						vec_imagesSize_pre.reserve(sfm_data_pre.GetViews().size());
						for (Views::const_iterator iter = sfm_data_pre.GetViews().begin();
							iter != sfm_data_pre.GetViews().end();
							++iter)
						{
							const View * v = iter->second.get();
							vec_fileNames_pre.push_back(stlplus::create_filespec(sfm_data_pre.s_root_path,
								v->s_Img_path));
							vec_imagesSize_pre.push_back(std::make_pair(v->ui_width, v->ui_height));
						}
					}
					//read descriptions read descriptions read descriptions read descriptions read descriptions read descriptions
				}
				else if (firstIterNext > firstIter + 1 && firstIterNext < group_count - 2)
				{
					matchBetweenGroups
					(
						firstIter,
						firstIterNext,
						matches_final_result_dir,
						matches_final_result_dir_next,
						regions_provider,
						regions_provider_next
					);
					//预读组数据赋值给待匹配组数据，再预读一组(firstIterNext+2)特征描述符数据进来
					matches_final_result_dir_next = matches_final_result_dir_pre;
					regions_provider_next = regions_provider_pre;
					vec_fileNames_next = vec_fileNames_pre;
					vec_imagesSize_next = vec_imagesSize_pre;

					//read descriptions read descriptions read descriptions read descriptions read descriptions read descriptions
					char temp_firstIterPre[2] = { ' ','\0' };
					temp_firstIterPre[0] = firstIterNext + 2 + 48;
					std::string str_firstIterPre = temp_firstIterPre;

					matches_final_result_dir_pre = sSfM_Data_FilenameDir_father + "DJI_" + str_firstIterPre + "_build/";
					if (matches_final_result_dir_pre.empty() || !stlplus::is_folder(matches_final_result_dir_pre))
					{
						std::cerr << "\nIt is an invalid output directory" << std::endl;
						return EXIT_FAILURE;
					}
					//---------------------------------------
					// Read SfM Scene (image view & intrinsics data)
					//---------------------------------------
					sfm_data_filename_pre = matches_final_result_dir_pre + "sfm_data.json";
					//SfM_Data sfm_data_pre;
					if (!Load(sfm_data_pre, sfm_data_filename_pre, ESfM_Data(VIEWS | INTRINSICS))) {
						std::cerr << std::endl
							<< "The input SfM_Data file \"" << sfm_data_filename_pre << "\" cannot be read." << std::endl;
						return EXIT_FAILURE;
					}
					//---------------------------------------
					// Load SfM Scene regions
					//---------------------------------------
					// Init the regions_type from the image describer file (used for image regions extraction)
					using namespace openMVG::features;
					// Load the corresponding view regions
					//std::shared_ptr<Regions_Provider> regions_provider_pre;
					// Default regions provider (load & store all regions in memory)
					regions_provider_pre = std::make_shared<Regions_Provider>();
					// Show the progress on the command line:
					//C_Progress_display progress_pre;

					if (!regions_provider_pre->load(sfm_data_pre, matches_final_result_dir_pre, regions_type, &progress_pre)) {
						std::cerr << "Invalid regions." << std::endl;
						return EXIT_FAILURE;
					}

					// Build some alias from SfM_Data Views data:
					// - List views as a vector of filenames & image sizes
					/*std::vector<std::string> vec_fileNames_pre;
					std::vector<std::pair<size_t, size_t>> vec_imagesSize_pre;*/
					{
						vec_fileNames_pre.reserve(sfm_data_pre.GetViews().size());
						vec_imagesSize_pre.reserve(sfm_data_pre.GetViews().size());
						for (Views::const_iterator iter = sfm_data_pre.GetViews().begin();
							iter != sfm_data_pre.GetViews().end();
							++iter)
						{
							const View * v = iter->second.get();
							vec_fileNames_pre.push_back(stlplus::create_filespec(sfm_data_pre.s_root_path,
								v->s_Img_path));
							vec_imagesSize_pre.push_back(std::make_pair(v->ui_width, v->ui_height));
						}
					}
					//read descriptions read descriptions read descriptions read descriptions read descriptions read descriptions
				}
				else if (firstIterNext == group_count - 2)
				{
					//将当前组和待匹配组进行匹配
					matchBetweenGroups
					(
						firstIter,
						firstIterNext,
						matches_final_result_dir,
						matches_final_result_dir_next,
						regions_provider,
						regions_provider_next
					);
					//预读组数据赋值给待匹配组数据
					matches_final_result_dir_next = matches_final_result_dir_pre;
					regions_provider_next = regions_provider_pre;
					vec_fileNames_next = vec_fileNames_pre;
					vec_imagesSize_next = vec_imagesSize_pre;
					//记得更新一些值
					firstIterNext++;
					//第一组和待匹配组进行匹配 matchBetweenGroups();
					matchBetweenGroups
					(
						firstIter,
						firstIterNext,
						matches_final_result_dir,
						matches_final_result_dir_next,
						regions_provider,
						regions_provider_next
					);
				}
				else
				{
					std::cout << " error indexing when matching between groups" << std::endl;
				}

				//组内自我匹配
				// If the matches already exists, reload them
				PairWiseMatches map_PutativesMatches_itself;
				std::string matches_result_filaname_itself = matches_final_result_dir + "/matches.putative_itself.bin";
				if (stlplus::file_exists(matches_result_filaname_itself))
				{
					if (!(Load(map_PutativesMatches_itself, matches_result_filaname_itself)))
					{
						std::cerr << "Cannot load input matches file";
						return EXIT_FAILURE;
					}
					std::cout << "\t PREVIOUS RESULTS LOADED;"
						<< " #pair: " << map_PutativesMatches_itself.size() << std::endl;
				}
				else
				{
					std::cout << "match for group " << firstIter << " has done!" << std::endl;

					matchForThisGroup(map_PutativesMatches_itself, firstIter, matches_final_result_dir, *regions_provider.get());

					//---------------------------------------
					//-- Export putative matches
					//---------------------------------------
					if (!Save(map_PutativesMatches_itself, std::string(matches_final_result_dir + "/matches.putative_itself.bin")))
					{
						std::cerr
							<< "Cannot save computed matches in: "
							<< std::string(matches_final_result_dir + "/matches.putative_itself.bin");
						return EXIT_FAILURE;
					}

					std::cout << "match for group " << firstIter << " itself has done!" << std::endl;
				}
			}
		}
		else if(firstIter == group_count - 3)
		{
			int firstIterNext = firstIter + 1;
			//读待匹配(firstIterNext)的一组特征描述符
			//read descriptions read descriptions read descriptions read descriptions read descriptions read descriptions
			char temp_firstIterNext[2] = { ' ','\0' };
			temp_firstIterNext[0] = firstIterNext + 48;
			const std::string str_firstIterNext = temp_firstIterNext;

			matches_final_result_dir_next = sSfM_Data_FilenameDir_father + "DJI_" + str_firstIterNext + "_build/";
			if (matches_final_result_dir_next.empty() || !stlplus::is_folder(matches_final_result_dir_next))
			{
				std::cerr << "\nIt is an invalid output directory" << std::endl;
				return EXIT_FAILURE;
			}
			//---------------------------------------
			// Read SfM Scene (image view & intrinsics data)
			//---------------------------------------
			sfm_data_filename_next = matches_final_result_dir_next + "sfm_data.json";
			if (!Load(sfm_data_next, sfm_data_filename_next, ESfM_Data(VIEWS | INTRINSICS))) {
				std::cerr << std::endl
					<< "The input SfM_Data file \"" << sfm_data_filename << "\" cannot be read." << std::endl;
				return EXIT_FAILURE;
			}
			//---------------------------------------
			// Load SfM Scene regions
			//---------------------------------------
			// Init the regions_type from the image describer file (used for image regions extraction)
			using namespace openMVG::features;
			// Default regions provider (load & store all regions in memory)
			regions_provider_next = std::make_shared<Regions_Provider>();

			if (!regions_provider_next->load(sfm_data_next, matches_final_result_dir_next, regions_type, &progress_next)) {
				std::cerr << std::endl << "Invalid regions." << std::endl;
				return EXIT_FAILURE;
			}

			// Build some alias from SfM_Data Views data:
			// - List views as a vector of filenames & image sizes
			{
				vec_fileNames_next.reserve(sfm_data_next.GetViews().size());
				vec_imagesSize_next.reserve(sfm_data_next.GetViews().size());
				for (Views::const_iterator iter = sfm_data_next.GetViews().begin();
					iter != sfm_data_next.GetViews().end();
					++iter)
				{
					const View * v = iter->second.get();
					vec_fileNames_next.push_back(stlplus::create_filespec(sfm_data_next.s_root_path,
						v->s_Img_path));
					vec_imagesSize_next.push_back(std::make_pair(v->ui_width, v->ui_height));
				}
			}
			//read descriptions read descriptions read descriptions read descriptions read descriptions read descriptions

			//读预读组(firstIterNext+1)的特征描述符数据
			//read descriptions read descriptions read descriptions read descriptions read descriptions read descriptions
			char temp_firstIterPre[2] = { ' ','\0' };
			temp_firstIterPre[0] = firstIterNext + 1 + 48;
			std::string str_firstIterPre = temp_firstIterPre;

			matches_final_result_dir_pre = sSfM_Data_FilenameDir_father + "DJI_" + str_firstIterPre + "_build/";
			if (matches_final_result_dir_pre.empty() || !stlplus::is_folder(matches_final_result_dir_pre))
			{
				std::cerr << "\nIt is an invalid output directory" << std::endl;
				return EXIT_FAILURE;
			}
			//---------------------------------------
			// Read SfM Scene (image view & intrinsics data)
			//---------------------------------------
			sfm_data_filename_pre = matches_final_result_dir_pre + "sfm_data.json";
			if (!Load(sfm_data_pre, sfm_data_filename_pre, ESfM_Data(VIEWS | INTRINSICS))) {
				std::cerr << std::endl
					<< "The input SfM_Data file \"" << sfm_data_filename_pre << "\" cannot be read." << std::endl;
				return EXIT_FAILURE;
			}
			//---------------------------------------
			// Load SfM Scene regions
			//---------------------------------------
			// Init the regions_type from the image describer file (used for image regions extraction)
			using namespace openMVG::features;
			// Default regions provider (load & store all regions in memory)
			regions_provider_pre = std::make_shared<Regions_Provider>();

			if (!regions_provider_pre->load(sfm_data_pre, matches_final_result_dir_pre, regions_type, &progress_pre)) {
				std::cerr << "Invalid regions." << std::endl;
				return EXIT_FAILURE;
			}

			// Build some alias from SfM_Data Views data:
			// - List views as a vector of filenames & image sizes
			{
				vec_fileNames_pre.reserve(sfm_data_pre.GetViews().size());
				vec_imagesSize_pre.reserve(sfm_data_pre.GetViews().size());
				for (Views::const_iterator iter = sfm_data_pre.GetViews().begin();
					iter != sfm_data_pre.GetViews().end();
					++iter)
				{
					const View * v = iter->second.get();
					vec_fileNames_pre.push_back(stlplus::create_filespec(sfm_data_pre.s_root_path,
						v->s_Img_path));
					vec_imagesSize_pre.push_back(std::make_pair(v->ui_width, v->ui_height));
				}
			}
			//read descriptions read descriptions read descriptions read descriptions read descriptions read descriptions
			//将当前组和待匹配组进行匹配
			matchBetweenGroups
			(
				firstIter,
				firstIterNext,
				matches_final_result_dir,
				matches_final_result_dir_next,
				regions_provider,
				regions_provider_next
			);
			//最后两组数据匹配
			matchBetweenGroups
			(
				firstIterNext,
				firstIterNext+1,
				matches_final_result_dir_next,
				matches_final_result_dir_pre,
				regions_provider_next,
				regions_provider_pre
			);
			//最后三组
			//firstIter
			//firstIterNext
			//firstIterNext+1

			//倒数第三组组内匹配
			{
				//组内自我匹配
				// If the matches already exists, reload them
				PairWiseMatches map_PutativesMatches_itself;
				std::string matches_result_filaname_itself = matches_final_result_dir + "/matches.putative_itself.bin";
				if (stlplus::file_exists(matches_result_filaname_itself))
				{
					if (!(Load(map_PutativesMatches_itself, matches_result_filaname_itself)))
					{
						std::cerr << "Cannot load input matches file";
						return EXIT_FAILURE;
					}
					std::cout << "\t PREVIOUS RESULTS LOADED;"
						<< " #pair: " << map_PutativesMatches_itself.size() << std::endl;
				}
				else
				{
					std::cout << "match for group " << firstIter << " has done!" << std::endl;

					matchForThisGroup(map_PutativesMatches_itself, firstIter, matches_final_result_dir, *regions_provider.get());

					//---------------------------------------
					//-- Export putative matches
					//---------------------------------------
					if (!Save(map_PutativesMatches_itself, std::string(matches_final_result_dir + "/matches.putative_itself.bin")))
					{
						std::cerr
							<< "Cannot save computed matches in: "
							<< std::string(matches_final_result_dir + "/matches.putative_itself.bin");
						return EXIT_FAILURE;
					}

					std::cout << "match for group " << firstIter << " itself has done!" << std::endl;
				}
			}
			//倒数第二组组内匹配
			//firstNext
			{
				//组内自我匹配
				// If the matches already exists, reload them
				PairWiseMatches map_PutativesMatches_itself;
				std::string matches_result_filaname_itself = matches_final_result_dir_next + "/matches.putative_itself.bin";
				if (stlplus::file_exists(matches_result_filaname_itself))
				{
					if (!(Load(map_PutativesMatches_itself, matches_result_filaname_itself)))
					{
						std::cerr << "Cannot load input matches file";
						return EXIT_FAILURE;
					}
					std::cout << "\t PREVIOUS RESULTS LOADED;"
						<< " #pair: " << map_PutativesMatches_itself.size() << std::endl;
				}
				else
				{
					std::cout << "match for group " << firstIterNext << " has done!" << std::endl;

					matchForThisGroup(map_PutativesMatches_itself, firstIterNext, matches_final_result_dir_next, *regions_provider_next.get());

					//---------------------------------------
					//-- Export putative matches
					//---------------------------------------
					if (!Save(map_PutativesMatches_itself, std::string(matches_final_result_dir_next + "/matches.putative_itself.bin")))
					{
						std::cerr
							<< "Cannot save computed matches in: "
							<< std::string(matches_final_result_dir_next + "/matches.putative_itself.bin");
						return EXIT_FAILURE;
					}

					std::cout << "match for group " << firstIterNext << " itself has done!" << std::endl;
				}
			}
			//倒数第一组组内匹配
			//firstIterNext + 1
			{
				//组内自我匹配
				// If the matches already exists, reload them
				PairWiseMatches map_PutativesMatches_itself;
				std::string matches_result_filaname_itself = matches_final_result_dir_pre + "/matches.putative_itself.bin";
				if (stlplus::file_exists(matches_result_filaname_itself))
				{
					if (!(Load(map_PutativesMatches_itself, matches_result_filaname_itself)))
					{
						std::cerr << "Cannot load input matches file";
						return EXIT_FAILURE;
					}
					std::cout << "\t PREVIOUS RESULTS LOADED;"
						<< " #pair: " << map_PutativesMatches_itself.size() << std::endl;
				}
				else
				{
					std::cout << "match for group " << firstIterNext + 1 << " has done!" << std::endl;

					matchForThisGroup(map_PutativesMatches_itself, firstIterNext + 1, matches_final_result_dir_pre, *regions_provider_pre.get());

					//---------------------------------------
					//-- Export putative matches
					//---------------------------------------
					if (!Save(map_PutativesMatches_itself, std::string(matches_final_result_dir_pre + "/matches.putative_itself.bin")))
					{
						std::cerr
							<< "Cannot save computed matches in: "
							<< std::string(matches_final_result_dir_pre + "/matches.putative_itself.bin");
						return EXIT_FAILURE;
					}

					std::cout << "match for group " << firstIterNext + 1 << " itself has done!" << std::endl;
				}
			}
			//预读组数据赋值给待匹配组数据
			matches_final_result_dir_next = matches_final_result_dir_pre;
			regions_provider_next = regions_provider_pre;
			vec_fileNames_next = vec_fileNames_pre;
			vec_imagesSize_next = vec_imagesSize_pre;
			//记得更新一些值
			firstIterNext++;
			//第一组和最后一组待匹配组进行匹配 matchBetweenGroups();
			matchBetweenGroups
			(
				firstIter,
				firstIterNext,
				matches_final_result_dir,
				matches_final_result_dir_next,
				regions_provider,
				regions_provider_next
			);
		}
		else 
		{
			std::cout << "error when indexing in the first layer data schedule!" << std::endl;
		}
	}

	std::cout << "match for all groups has done!" << std::endl;
	return EXIT_SUCCESS;
}
int computeMatches::showMatchesOnImage() 
{
	char temp_i[2] = { ' ','\0' };
	temp_i[0] = 48;
	const std::string str_i = temp_i;
	std::string imageDir = sMatchesOutputDir_father + "DJI_" + str_i + "_build/";
	return EXIT_SUCCESS;
}