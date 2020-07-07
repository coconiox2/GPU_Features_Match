#pragma once
#ifndef _COMPUTE_FEATURES_CU_H_
#define _COMPUTE_FEATURES_CU_H_



#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"
#include <string>


namespace computeMatches {
	//Defines the path and name of the read and output files
	const std::string sSfM_Data_FilenameDir = stlplus::folder_up(imageInputDir, 4) +
		"/imageData/360_school/build/";

	const std::string sSfM_Data_Filename = sSfM_Data_FilenameDir + "sfm_data.json";

	const std::string sMatchesOutputDir = stlplus::folder_up(imageInputDir, 4) +
		"/imageData/360_school/build/";
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
		int computeMatches();
	};

}//namespace computeMatches



#endif // !_COMPUTE_FEATURES_CU_H_
