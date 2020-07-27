#pragma once
#ifndef _COMPUTE_FEATURES_CU_H_
#define _COMPUTE_FEATURES_CU_H_


#include <Eigen/Dense>
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"
#include <string>


namespace computeMatches {

	const int group_count = 6;
	const int block_count_per_group = 2;
	const int image_count_per_block = 3;
	const int image_count_per_group = 6;
	const int descriptionDimension = 128;
	//Defines the path and name of the read and output files

	const std::string sSfM_Data_FilenameDir_father = stlplus::folder_up(imageInputDir, 4) +
		"/imageData/tianjin/";

	//const std::string sSfM_Data_Filename = sSfM_Data_FilenameDir + "sfm_data.json";

	const std::string sMatchesOutputDir_father = stlplus::folder_up(imageInputDir, 4) +
		"/imageData/tianjin/";

	struct HashedDescription {
		// Hash code generated by the primary hashing function.
		stl::dynamic_bitset hash_code;
		
		// Each bucket_ids[x] = y means the descriptor belongs to bucket y in bucket
		// group x.
		std::vector<uint16_t> bucket_ids;
	};

	struct HashedDescriptions {
		// The hash information.
		std::vector<HashedDescription> hashed_desc;
		 
		using Bucket = std::vector<int>;
		// buckets[bucket_group][bucket_id] = bucket (container of description ids).
		std::vector<std::vector<Bucket>> buckets;
	};

	Eigen::VectorXf zero_mean_descriptor;
	const bool computeMatchesWithGPU = true;
	
	class ComputeMatches {
	public:
		//geometric model
		//fundamental matrix、essential matrix、homography matrix
		//angular essential matrix、ortho essential matrix
		enum EGeometricModel
		{
			FUNDAMENTAL_MATRIX = 0,
			ESSENTIAL_MATRIX = 1,
			HOMOGRAPHY_MATRIX = 2,
			ESSENTIAL_MATRIX_ANGULAR = 3,
			ESSENTIAL_MATRIX_ORTHO = 4
		};
		//匹配对模式
		//详细匹配对、临近匹配对、从文件中读取的匹配对
		enum EPairMode
		{
			PAIR_EXHAUSTIVE = 0,
			PAIR_CONTIGUOUS = 1,
			PAIR_FROM_FILE = 2
		};
		/// Compute corresponding features between a series of views:
		/// - Load view images description (regions: features & descriptors)
		/// - Compute putative local feature matches (descriptors matching)
		/// - Compute geometric coherent feature matches (robust model estimation from putative matches)
		/// - Export computed data
		void test();
		void computeZeroMeanDescriptors(Eigen::VectorXf &zero_mean_descriptor);
		int computeHashes();
		int computeMatches();
	};

}//namespace computeMatches



#endif // !_COMPUTE_FEATURES_CU_H_
