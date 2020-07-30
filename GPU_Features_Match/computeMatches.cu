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
#include <iostream>
//CUDA V 10.2
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

using namespace computeMatches;
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
//extern "C"
//void hash_gen
//(
//	//右式B的列数
//	int mat_I_cols_count,
//	//左式A的行数列数 右式B的行数均为descriptionDimension
//	int descriptionDimension,
//	//左式A
//	float *primary_hash_projection_data_device,
//	//右式B
//	const float *descriptions_GPU,
//	//结果C
//	float *hash_base_GPU
//)
//{
//	/*dim3 threads(1, 1);
//	dim3 grid(1, 1);*/
//	// 创建并初始化 CUBLAS 库对象
//	cublasHandle_t handle;
//	int status = cublasCreate(&handle);
//	if (status != CUBLAS_STATUS_SUCCESS)
//	{
//		if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
//			std::cout << "CUBLAS 对象实例化出错" << std::endl;
//		}
//		getchar();
//
//	}
//	// 同步函数
//	cudaThreadSynchronize();
//	// 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
//	float a = 1; float b = 0;
//	// 矩阵相乘。该函数必然将数组解析成列优先数组
//
//	status = cublasSgemm
//	(
//		//A(primary_hash_proejection)
//		//B(description_Mat)
//		//C(result)
//		handle, //blas库对象
//		CUBLAS_OP_N, //矩阵A不转置
//		CUBLAS_OP_N, //矩阵B不转置
//		descriptionDimension, //矩阵A、C的行数,也即结果的行数
//		mat_I_cols_count, //矩阵B、C的列数，也即结果的列数
//		descriptionDimension, //矩阵A的列数或者B的行数
//		&a,  //alpha的值
//		primary_hash_projection_data_device, //左矩阵A
//		descriptionDimension, //A的leading dimension,以列为主就填行数
//		descriptions_GPU, //右矩阵 B
//		descriptionDimension, //矩阵B的leading dimension,以列为主就填行数
//		&b,             //beta的值
//		hash_base_GPU, //结果矩阵C
//		descriptionDimension//结果矩阵C的leading dimension,以列为主就填行数
//	);
//	cublasDestroy(handle);
//}
//
//extern "C"
//void determine_buket_index_for_each_group
//(
//	//计算结果C
//	float *secondary_projection_GPU,
//	//左式A
//	float *secondary_hash_projection_j,
//	//右式B
//	const float *descriptions_GPU,
//	//左式A的行数  10
//	int secondary_rows,
//	//左式A的列数 128
//	int secondary_cols,
//	//右式B的列数 
//	int descrptions_cols
//	//右式B的行数就是descriptionDimension
//)
//{
//	/*dim3 threads(1, 1);
//	dim3 grid(1, 1);*/
//	// 创建并初始化 CUBLAS 库对象
//	cublasHandle_t handle;
//	int status = cublasCreate(&handle);
//	if (status != CUBLAS_STATUS_SUCCESS)
//	{
//		if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
//			std::cout << "CUBLAS 对象实例化出错" << std::endl;
//		}
//		getchar();
//
//	}
//	// 同步函数
//	cudaThreadSynchronize();
//	// 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
//	float a = 1; float b = 0;
//	// 矩阵相乘。该函数必然将数组解析成列优先数组
//
//	status = cublasSgemm
//	(
//		//A(primary_hash_proejection)
//		//B(description_Mat)
//		//C(result)
//		handle, //blas库对象
//		CUBLAS_OP_N, //矩阵A不转置
//		CUBLAS_OP_N, //矩阵B转置
//		secondary_rows, //矩阵A、C的行数,也即结果的行数
//		descrptions_cols, //矩阵B、C的列数，也即结果的列数
//		descriptionDimension, //矩阵A的列数或者B的行数
//		&a,  //alpha的值
//		secondary_hash_projection_j, //左矩阵A
//		secondary_rows, //A的leading dimension,以列为主就填行数
//		descriptions_GPU, //右矩阵 B
//		descriptionDimension, //矩阵B的leading dimension,以列为主就填行数
//		&b,             //beta的值
//		secondary_projection_GPU, //结果矩阵C
//		secondary_rows//结果矩阵C的leading dimension
//	);
//	cublasDestroy(handle);
//}