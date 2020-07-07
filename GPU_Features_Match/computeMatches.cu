#include "computeMatchesCU.h"



//openMVG
//#include "openMVG/graph/graph.hpp"
//#include "openMVG/features/akaze/image_describer_akaze.hpp"
//#include "openMVG/features/descriptor.hpp"
//#include "openMVG/features/feature.hpp"
//#include "openMVG/matching/indMatch.hpp"
//#include "openMVG/matching/indMatch_utils.hpp"
//#include "openMVG/matching_image_collection/Matcher_Regions.hpp"
//#include "openMVG/matching_image_collection/Cascade_Hashing_Matcher_Regions.hpp"
//#include "openMVG/matching_image_collection/GeometricFilter.hpp"
//#include "openMVG/sfm/pipelines/sfm_features_provider.hpp"
//#include "openMVG/sfm/pipelines/sfm_regions_provider.hpp"
//#include "openMVG/sfm/pipelines/sfm_regions_provider_cache.hpp"
//#include "openMVG/matching_image_collection/F_ACRobust.hpp"
//#include "openMVG/matching_image_collection/E_ACRobust.hpp"
//#include "openMVG/matching_image_collection/E_ACRobust_Angular.hpp"
//#include "openMVG/matching_image_collection/Eo_Robust.hpp"
//#include "openMVG/matching_image_collection/H_ACRobust.hpp"
//#include "openMVG/matching_image_collection/Pair_Builder.hpp"
//#include "openMVG/matching/pairwiseAdjacencyDisplay.hpp"
//#include "openMVG/sfm/sfm_data.hpp"
//#include "openMVG/sfm/sfm_data_io.hpp"
//#include "openMVG/stl/stl.hpp"
//#include "openMVG/system/timer.hpp"
//
//#include "third_party/cmdLine/cmdLine.h"
//#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"
//
//CUDA V 10.2
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
//
//#include <cstdlib>
//#include <iostream>
//#include <memory>
//#include <string>
//
//using namespace openMVG;
//using namespace openMVG::matching;
//using namespace openMVG::robust;
//using namespace openMVG::sfm;
//using namespace openMVG::matching_image_collection;
//using namespace std;
extern "C" int testCUDACPP(int a,int b) {
	int c = a + b;
	return c;
}

__global__ void Mul(float* descriptionsMat_data_device, float* primary_hash_projection_data_device, float* mat_result, int m_result_rows, int m_result_cols) {
	
}