
#include "computeMatchesCU.h"
#include "Cascade_Hashing_Matcher_Regions_GPU.hpp"
#include "cascade_hasher_GPU.hpp"

//openMVG
#include "openMVG/graph/graph.hpp"
#include "openMVG/features/akaze/image_describer_akaze.hpp"
#include "openMVG/features/descriptor.hpp"
#include "openMVG/features/feature.hpp"
#include "openMVG/matching/indMatch.hpp"
#include "openMVG/matching/indMatch_utils.hpp"
#include "openMVG/matching_image_collection/Matcher_Regions.hpp"
#include "openMVG/matching_image_collection/Cascade_Hashing_Matcher_Regions.hpp"
#include "openMVG/matching_image_collection/GeometricFilter.hpp"
#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
#include "openMVG/sfm/pipelines/sfm_regions_provider.hpp"
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
using namespace openMVG::matching;
using namespace openMVG::robust;
using namespace openMVG::sfm;
using namespace openMVG::matching_image_collection;
using namespace std;

#define max_descriptions_num_per_image 40000	//（4*keypointsNum）

namespace openMVG {
	namespace matching_image_collection {
		class Cascade_Hash_Generate {
		public:
			
			using Map_vectorT = std::map<IndexT, std::vector<IndexT>>;
			
			template <typename ScalarT>
			void Hash
			(
				//offer openMVG::features & openMVG::descriptor
				const sfm::Regions_Provider & regions_provider,
				std::map<IndexT, HashedDescriptions> & hashed_base_,
				//pairs of views which need to be match and calculate
				const Pair_Set & pairs,
				Map_vectorT map_Pairs,
				std::set<IndexT> used_index,
				const int firstHash = 0,
				C_Progress * my_progress_bar
			) 
			{
				
				

				//// Collect used view indexes
				//std::set<IndexT> used_index;
				//// Sort pairs according the first index to minimize later memory swapping
				////std::map use red&black tree to sort it's members automatically
				////IndexT <--> vector,There are multiple views for each view to match
				//using Map_vectorT = std::map<IndexT, std::vector<IndexT>>;
				//Map_vectorT map_Pairs;
				//for (const auto & pair_idx : pairs)
				//{
				//	map_Pairs[pair_idx.first].push_back(pair_idx.second);
				//	used_index.insert(pair_idx.first);
				//	used_index.insert(pair_idx.second);
				//}

				if (!my_progress_bar)
					my_progress_bar = &C_Progress::dummy();
				my_progress_bar->restart(used_index.size(), "\n- hash generating -\n");

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

				//std::map<IndexT, HashedDescriptions> hashed_base_;

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

				openMVG::matching::RTOC myRTOC;
				//在这里把zero_mean_descriptor和primary_hash_projection_上传到device上去
				{

					const float *primary_hash_projection_data_temp = cascade_hasher.primary_hash_projection_.data();
					const float *secondary_hash_projection_data_temp = cascade_hasher.secondary_hash_projection_.data();

					float *primary_hash_projection_data_temp_1 = (float*)malloc(cascade_hasher.primary_hash_projection_.rows()*cascade_hasher.primary_hash_projection_.cols() * sizeof(float));
					float *secondary_hash_projection_data_temp_1 = (float*)malloc(cascade_hasher.secondary_hash_projection_.rows()*cascade_hasher.secondary_hash_projection_.cols() * sizeof(float));

					//行优先转化为列优先
					myRTOC.cToR(primary_hash_projection_data_temp, cascade_hasher.primary_hash_projection_.rows(), cascade_hasher.primary_hash_projection_.cols(), primary_hash_projection_data_temp_1);
					myRTOC.cToR(secondary_hash_projection_data_temp, cascade_hasher.secondary_hash_projection_.rows(), cascade_hasher.secondary_hash_projection_.cols(), secondary_hash_projection_data_temp_1);
					const float *primary_hash_projection_data = const_cast<const float*>(primary_hash_projection_data_temp_1);
					const float *secondary_hash_projection_data = const_cast<const float*>(secondary_hash_projection_data_temp_1);
					int primary_hash_projection_size = (cascade_hasher.primary_hash_projection_.rows()) * (cascade_hasher.primary_hash_projection_.cols());
					int secondary_hash_projection_size = (cascade_hasher.secondary_hash_projection_.rows()) * (cascade_hasher.secondary_hash_projection_.cols());
					float *primary_hash_projection_data_device,
						*secondary_hash_projection_data_device;

					cudaMalloc((void **)&primary_hash_projection_data_device, sizeof(float) * primary_hash_projection_size);
					cudaMalloc((void **)&secondary_hash_projection_data_device, sizeof(float) * secondary_hash_projection_size);

					// 将矩阵数据传递进 显存 中已经开辟好了的空间
					cudaMemcpy(primary_hash_projection_data_device, primary_hash_projection_data, sizeof(float) * primary_hash_projection_size, cudaMemcpyHostToDevice);
					cudaMemcpy(secondary_hash_projection_data_device, secondary_hash_projection_data, sizeof(float) * secondary_hash_projection_size, cudaMemcpyHostToDevice);
					// 同步函数
					cudaThreadSynchronize();
				}
				

				// Index the input regions
				//std::vector<Eigen::Map<BaseMat>> vec_Mat_I;
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
					//descriptionsMat = descriptions.template cast<float>();
					mat_I = mat_I.template cast<float>();
					//vec_Mat_I.push_back(mat_I);
				}
				//处理好描述符数据传到GPU上
				{
					// Allocate space for hash codes.
					const typename MatrixT::Index nbDescriptions = descriptions.rows();
					//hashed_descriptions.hashed_desc.resize(nbDescriptions);


					//整合所有的描述符放在一个矩阵里传到device
					Eigen::MatrixXf descriptionsMat;
					descriptionsMat = descriptions.template cast<float>();
					for (int k = 0; k < descriptionsMat.rows(); k++) {
						descriptionsMat.row(k) -= zero_mean_descriptor;
					}
					int descriptionsMatSize = descriptionsMat.rows() * descriptionsMat.cols();
					const float *descriptionsMat_data_temp = descriptionsMat.data();
					float *descriptionsMat_data_temp_1 = (float*)malloc(descriptions.rows()*descriptions.cols() * sizeof(float));
					myRTOC.cToR(descriptionsMat_data_temp, descriptionsMat.rows(), descriptionsMat.cols(), descriptionsMat_data_temp_1);
					const float *descriptionsMat_data = const_cast<const float*> (descriptionsMat_data_temp_1);
					float *descriptionsMat_data_device;
					cudaMalloc((void **)&descriptionsMat_data_device, sizeof(float) * descriptions_size);
					// 将矩阵数据传递进 显存 中已经开辟好了的空间
					cudaMemcpy(descriptionsMat_data_device, descriptionsMat_data, sizeof(float) * descriptions_size, cudaMemcpyHostToDevice);
					// 同步函数
					cudaThreadSynchronize();
					//该写第二层数据调度策略了

				}


				//顺便在device上为hashed_base_申请好空间
				float *hashed_base_device;
				cudaMalloc((void **)&hashed_base_device, sizeof(float) * max_descriptions_num_per_image);
				cascade_hasher.CreateHashedDescriptions(vec_Mat_I, zero_mean_descriptor, hashed_base_);
			}
		};
	}//namespace openMVG
}//namespace matching_image_collection

extern "C" int testCUDACPP(int a, int b);

//#include "third_party/eigen/eigen/src/Core/util/Macros.h"
void computeMatches::ComputeMatches::test() {
	int a = 1;
	int b = 2;
	int c = -1;
	c = testCUDACPP(a, b);
	printf("testCUDACPP result:%d", c);
	printf("cpp test success\n");
}
void computeMatches::ComputeMatches::computeZeroMeanDescriptors(Eigen::VectorXf &zero_mean_descriptor)
{
	int imgCount = group_count*block_count_per_group*image_count_per_block;
	//用于保存每组数据计算得到的零和平均值，最后再平均为zero_mean_descriptors
	Eigen::MatrixXf zero_descriptor;
	zero_descriptor.resize(imgCount , descriptionDimension);
	zero_descriptor.fill(0.0f);

	//初始化cascade hasher
	CascadeHasher zeroCascadeHasher;
	zeroCascadeHasher.Init(descriptionDimension);

	std::string sSfM_Data_Filename;
	std::string sMatchesOutputDir;
	char temp_i[2] = { ' ','\0' };
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
		//compute begin
		using BaseMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
		Eigen::MatrixXf matForZeroMean;
		for (int j = 0; j < image_count_per_group; j++) {
			const IndexT I = j;
			const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
			const float * tabI =
				reinterpret_cast<const float*>(regionsI->DescriptorRawData());
			const size_t dimension = regionsI->DescriptorLength();
			if (i == 0)
			{
				//Each row of the matrix is the size of a descriptor
				matForZeroMean.resize(image_count_per_group, dimension);
				matForZeroMean.fill(0.0f);
			}
			if (regionsI->RegionCount() > 0)
			{
				Eigen::Map<BaseMat> mat_I((float*)tabI, regionsI->RegionCount(), dimension);
				//GPU parallel here may be slower
				//return descriptions.template cast<float>().colwise().mean();
				matForZeroMean.row(i) = CascadeHasher::GetZeroMeanDescriptor(mat_I);
			}
			zero_descriptor.row(i) = CascadeHasher::GetZeroMeanDescriptor(matForZeroMean);
		}
		zero_mean_descriptor = CascadeHasher::GetZeroMeanDescriptor(zero_descriptor);
	}
			
	// Compute the zero mean descriptor that will be used for hashing (one for all the image regions)
	// A vector of undetermined size but with a value of float data
	{
		// A matrix of float type whose size is undetermined
		Eigen::MatrixXf matForZeroMean;
		
		for (int i = 0; i < imgCount; ++i)
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
}
int computeMatches::ComputeMatches::computeMatches()
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
	//分组读入图片数据之前就先把GPU上常用的空间先申请好
	//需要上传的一直用的数据也都上传
	CascadeHasher myCascadeHasher;
	myCascadeHasher.Init(128);

	//分组读入图片数据，1个sfm_data就是一组的图像数据
	//这里先计算哈希值
	for (int i = 0; i < group_count; i++) {
		char temp_i[2] = { ' ','\0' };
		temp_i[0] = i + 48;
		const std::string str_i = temp_i;

		if (i == 0) {
			sSfM_Data_Filename_hash = sSfM_Data_FilenameDir_father + "DJI_" + str_i + "_build/" + "sfm_data.json";
			sMatchesOutputDir_hash = sMatchesOutputDir_father + "DJI_" + str_i + "_build/";

			temp_i[0] = i + 1 + 48;
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

					std::cout << std::endl << " - PUTATIVE MATCHES - " << std::endl;
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
								collectionMatcher.reset(new Cascade_Hashing_Matcher_Regions_GPU(fDistRatio));
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
								////////////////////////////////////////////////////////////////////////////////////////
								//
								////注意每次读之前要把pairs(需要匹配的图像对)重新写好
								///////////////////////////////////////////////////////////////////////////////////////////
							case PAIR_EXHAUSTIVE: pairs = exhaustivePairs(sfm_data_hash.GetViews().size()); break;
							case PAIR_CONTIGUOUS: pairs = contiguousWithOverlap(sfm_data_hash.GetViews().size(), iMatchingVideoMode); break;
							case PAIR_FROM_FILE:
								if (!loadPairs(sfm_data_hash.GetViews().size(), sPredefinedPairList, pairs))
								{
									return EXIT_FAILURE;
								}
								break;
							}
							// Photometric matching of putative pairs
							
							
							
							
							//先修改好used_index
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

							
							std::map<IndexT, HashedDescriptions> hashed_base_;

							CascadeHasher myCascadeHasher;
							myCascadeHasher.Init(128);

							openMVG::matching_image_collection::Cascade_Hash_Generate sCascade_Hash_Generate;
							//第二层数据调度策略 CPU内存 <--> GPU内存
							using BaseMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
							//当前块数据
							std::vector<Eigen::Map<BaseMat>> vec_Mat_I;
							std::vector <int> mat_I_rows;
							mat_I_rows.resize(image_count_per_block);

							const float *mat_I_point_array[image_count_per_block];

							int vec_Mat_I_size = 0;
							//预读块数据
							std::vector<Eigen::Map<BaseMat>> vec_Mat_I_pre;
							std::vector <int> mat_I_pre_rows;
							mat_I_pre_rows.resize(image_count_per_block);

							const float *mat_I_pre_point_array[image_count_per_block];

							int vec_Mat_I_pre_size = 0;

							for (int secondIter = 0; secondIter < block_count_per_group; secondIter++) {
							//处理每一块的数据，保证大概能把GPU内存塞满(或许应该多次试验看哪一组实验效果最好)
								{
									if (secondIter == 0) {
										vec_Mat_I.resize(image_count_per_block);
										
										int hashed_base_size = 0;
										//处理块内每一张图片的数据
										for (int m = 0; m < image_count_per_block; ++m)
										{
											std::set<IndexT>::const_iterator iter = used_index.begin();
											std::advance(iter, m);
											const IndexT I = *iter;
											const std::shared_ptr<features::Regions> regionsI = (*regions_provider.get()).get(I);
											const float * tabI =
												reinterpret_cast<const float*>(regionsI->DescriptorRawData());
											const size_t dimension = regionsI->DescriptorLength();

											Eigen::Map<BaseMat> mat_I((float*)tabI, regionsI->RegionCount(), dimension);
											//descriptionsMat = descriptions.template cast<float>();
											//mat_I = mat_I.template cast<float>();
											
											Eigen::MatrixXf descriptionsMat;
											descriptionsMat = mat_I.template cast<float>();
											for (int k = 0; k < descriptionsMat.rows(); k++) {
												descriptionsMat.row(k) -= computeMatches::zero_mean_descriptor;
											}
											const float *descriptionsMat_data_temp = descriptionsMat.data();
											float *descriptionsMat_data_temp_1 = (float*)malloc(descriptionsMat.rows()*descriptionsMat.cols() * sizeof(float));
											openMVG::matching::RTOC myRTOC;
											myRTOC.cToR(descriptionsMat_data_temp, descriptionsMat.rows(), descriptionsMat.cols(), descriptionsMat_data_temp_1);
											//const float *descriptionsMat_data = const_cast<const float*> (descriptionsMat_data_temp_1);
											mat_I_point_array[m] = const_cast<const float*> (descriptionsMat_data_temp_1);

											mat_I_rows.push_back(mat_I.rows());
											hashed_base_size += mat_I.rows();
											vec_Mat_I.push_back(mat_I);
										}
										//把vec_Mat_I上传，并为hashed_base_申请空间
										//GPU上存放描述符的指针
										{
											float *vec_Mat_I_device;
											cudaMalloc((void **)&vec_Mat_I_device, sizeof(float) * hashed_base_size * descriptionDimension);
											 float* vec_Mat_I_data = vec_Mat_I.data();
											cudaMemcpy(vec_Mat_I_device, descriptionsMat_data, sizeof(float) * descriptions_size, cudaMemcpyHostToDevice);
										}
										
										//把vec_Mat_I_pre上传
									}
									else if (secondIter > 0 && secondIter < block_count_per_group - 1) {

									}
									else if (secondIter == block_count_per_group) {

									}
									else {
										cout << "error in the second data exchange schedule! \n" << endl;
										return EXIT_FAILURE;
									}
								}
							}
							sCascade_Hash_Generate.Hash<float>((*regions_provider.get()), hashed_base_, pairs, map_Pairs, used_index, 1, &progress);
							//将得到的哈希值hashed_base_传入到match里面去进行匹配，不过下面这一段应该放在下一个for循环(match的数据调度策略)里面去做
							collectionMatcher->Match(regions_provider, pairs, map_PutativesMatches, &progress);
							//---------------------------------------
							//-- Export putative matches
							//---------------------------------------
							if (!Save(map_PutativesMatches, std::string(sMatchesOutputDir_hash + "/matches.putative.bin")))
							{
								std::cerr
									<< "Cannot save computed matches in: "
									<< std::string(sMatchesOutputDir_hash + "/matches.putative.bin");
								return EXIT_FAILURE;
							}
						}
						std::cout << "Task (Regions Matching) done in (s): " << timer.elapsed() << std::endl;
					}
					//-- export putative matches Adjacency matrix
					PairWiseMatchingToAdjacencyMatrixSVG(vec_fileNames.size(),
						map_PutativesMatches,
						stlplus::create_filespec(sMatchesOutputDir_hash, "PutativeAdjacencyMatrix", "svg"));
					//-- export view pair graph once putative graph matches have been computed
					{
						std::set<IndexT> set_ViewIds;
						std::transform(sfm_data_hash.GetViews().begin(), sfm_data_hash.GetViews().end(),
							std::inserter(set_ViewIds, set_ViewIds.begin()), stl::RetrieveKey());
						graph::indexedGraph putativeGraph(set_ViewIds, getPairs(map_PutativesMatches));
						graph::exportToGraphvizData(
							stlplus::create_filespec(sMatchesOutputDir_hash, "putative_matches"),
							putativeGraph);
					}

					//---------------------------------------
					// b. Geometric filtering of putative matches
					//    - AContrario Estimation of the desired geometric model
					//    - Use an upper bound for the a contrario estimated threshold
					//---------------------------------------

					std::unique_ptr<ImageCollectionGeometricFilter> filter_ptr(
						new ImageCollectionGeometricFilter(&sfm_data_hash, regions_provider));

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
								GeometricFilter_ESphericalMatrix_AC_Angular(4.0, imax_iteration),
								map_PutativesMatches, bGuided_matching);
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
						}

						//---------------------------------------
						//-- Export geometric filtered matches
						//---------------------------------------
						if (!Save(map_GeometricMatches,
							std::string(sMatchesOutputDir_hash + "/" + sGeometricMatchesFilename)))
						{
							std::cerr
								<< "Cannot save computed matches in: "
								<< std::string(sMatchesOutputDir_hash + "/" + sGeometricMatchesFilename);
							return EXIT_FAILURE;
						}

						std::cout << "Task done in (s): " << timer.elapsed() << std::endl;

						//-- export Adjacency matrix
						std::cout << "\n Export Adjacency Matrix of the pairwise's geometric matches"
							<< std::endl;
						PairWiseMatchingToAdjacencyMatrixSVG(vec_fileNames.size(),
							map_GeometricMatches,
							stlplus::create_filespec(sMatchesOutputDir_hash, "GeometricAdjacencyMatrix", "svg"));

						//-- export view pair graph once geometric filter have been done
						{
							std::set<IndexT> set_ViewIds;
							std::transform(sfm_data_hash.GetViews().begin(), sfm_data_hash.GetViews().end(),
								std::inserter(set_ViewIds, set_ViewIds.begin()), stl::RetrieveKey());
							graph::indexedGraph putativeGraph(set_ViewIds, getPairs(map_GeometricMatches));
							graph::exportToGraphvizData(
								stlplus::create_filespec(sMatchesOutputDir_hash, "geometric_matches"),
								putativeGraph);
						}
					}
				}
				//
				//---------------------------------------
				// 把pre赋值给当前组数据，并读入第三组的数据作为pre
				//---------------------------------------
				sSfM_Data_Filename_hash = sSfM_Data_Filename_hash_pre;
				sMatchesOutputDir_hash = sMatchesOutputDir_hash_pre;
				sfm_data_hash = sfm_data_hash_pre;

				temp_i[0] = i + 2 + 48;
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
		else if(i>0 && i<group_count-1)
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

				std::cout << std::endl << " - PUTATIVE MATCHES - " << std::endl;
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
							collectionMatcher.reset(new Cascade_Hashing_Matcher_Regions_GPU(fDistRatio));
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
							////////////////////////////////////////////////////////////////////////////////////////
							//
							//
							///////////////////////////////////////////////////////////////////////////////////////////
						case PAIR_EXHAUSTIVE: pairs = exhaustivePairs(sfm_data_hash.GetViews().size()); break;
						case PAIR_CONTIGUOUS: pairs = contiguousWithOverlap(sfm_data_hash.GetViews().size(), iMatchingVideoMode); break;
						case PAIR_FROM_FILE:
							if (!loadPairs(sfm_data_hash.GetViews().size(), sPredefinedPairList, pairs))
							{
								return EXIT_FAILURE;
							}
							break;
						}
						// Photometric matching of putative pairs
						//GPU Parallel here 
						//这里需要新写一个哈希码生成的类成员函数，每次读2个组进去到cascade_hasher_GPU(内存)里面，
						//在cascade_hasher_GPU每次读2个块到GPU内存里
						//
						//
						//这里每次读3个组进去到cascade_hasher_GPU(内存)里面，在cascade_hasher_GPU每次读3个块到GPU内存里
						//注意每次读之前要把pairs(需要匹配的图像对)重新写好
						collectionMatcher->Match(regions_provider, pairs, map_PutativesMatches, &progress);
						//---------------------------------------
						//-- Export putative matches
						//---------------------------------------
						if (!Save(map_PutativesMatches, std::string(sMatchesOutputDir_hash + "/matches.putative.bin")))
						{
							std::cerr
								<< "Cannot save computed matches in: "
								<< std::string(sMatchesOutputDir_hash + "/matches.putative.bin");
							return EXIT_FAILURE;
						}
					}
					std::cout << "Task (Regions Matching) done in (s): " << timer.elapsed() << std::endl;
				}
				//-- export putative matches Adjacency matrix
				PairWiseMatchingToAdjacencyMatrixSVG(vec_fileNames.size(),
					map_PutativesMatches,
					stlplus::create_filespec(sMatchesOutputDir_hash, "PutativeAdjacencyMatrix", "svg"));
				//-- export view pair graph once putative graph matches have been computed
				{
					std::set<IndexT> set_ViewIds;
					std::transform(sfm_data_hash.GetViews().begin(), sfm_data_hash.GetViews().end(),
						std::inserter(set_ViewIds, set_ViewIds.begin()), stl::RetrieveKey());
					graph::indexedGraph putativeGraph(set_ViewIds, getPairs(map_PutativesMatches));
					graph::exportToGraphvizData(
						stlplus::create_filespec(sMatchesOutputDir_hash, "putative_matches"),
						putativeGraph);
				}

				//---------------------------------------
				// b. Geometric filtering of putative matches
				//    - AContrario Estimation of the desired geometric model
				//    - Use an upper bound for the a contrario estimated threshold
				//---------------------------------------

				std::unique_ptr<ImageCollectionGeometricFilter> filter_ptr(
					new ImageCollectionGeometricFilter(&sfm_data_hash, regions_provider));

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
							GeometricFilter_ESphericalMatrix_AC_Angular(4.0, imax_iteration),
							map_PutativesMatches, bGuided_matching);
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
					}

					//---------------------------------------
					//-- Export geometric filtered matches
					//---------------------------------------
					if (!Save(map_GeometricMatches,
						std::string(sMatchesOutputDir_hash + "/" + sGeometricMatchesFilename)))
					{
						std::cerr
							<< "Cannot save computed matches in: "
							<< std::string(sMatchesOutputDir_hash + "/" + sGeometricMatchesFilename);
						return EXIT_FAILURE;
					}

					std::cout << "Task done in (s): " << timer.elapsed() << std::endl;

					//-- export Adjacency matrix
					std::cout << "\n Export Adjacency Matrix of the pairwise's geometric matches"
						<< std::endl;
					PairWiseMatchingToAdjacencyMatrixSVG(vec_fileNames.size(),
						map_GeometricMatches,
						stlplus::create_filespec(sMatchesOutputDir_hash, "GeometricAdjacencyMatrix", "svg"));

					//-- export view pair graph once geometric filter have been done
					{
						std::set<IndexT> set_ViewIds;
						std::transform(sfm_data_hash.GetViews().begin(), sfm_data_hash.GetViews().end(),
							std::inserter(set_ViewIds, set_ViewIds.begin()), stl::RetrieveKey());
						graph::indexedGraph putativeGraph(set_ViewIds, getPairs(map_GeometricMatches));
						graph::exportToGraphvizData(
							stlplus::create_filespec(sMatchesOutputDir_hash, "geometric_matches"),
							putativeGraph);
					}
				}
			}
			
			//处理完当前组数据后，再执行下面几行代码，把第二组的数据赋值给第一组，再从磁盘上读取一组数据到内存里来放在pre里
			sSfM_Data_Filename_hash = sSfM_Data_Filename_hash_pre;
			sMatchesOutputDir_hash = sMatchesOutputDir_hash_pre;
			sfm_data_hash = sfm_data_hash_pre;

			temp_i[0] = i + 2 + 48;
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
		else if (i == (group_count - 1)) {
			
		}
		else {
			cout << "error in first data exchange schedule! \n" << endl;
			return EXIT_FAILURE;
		}
		
	}

	//这里计算匹配
}