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
//	//��ʽB������
//	int mat_I_cols_count,
//	//��ʽA���������� ��ʽB��������ΪdescriptionDimension
//	int descriptionDimension,
//	//��ʽA
//	float *primary_hash_projection_data_device,
//	//��ʽB
//	const float *descriptions_GPU,
//	//���C
//	float *hash_base_GPU
//)
//{
//	/*dim3 threads(1, 1);
//	dim3 grid(1, 1);*/
//	// ��������ʼ�� CUBLAS �����
//	cublasHandle_t handle;
//	int status = cublasCreate(&handle);
//	if (status != CUBLAS_STATUS_SUCCESS)
//	{
//		if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
//			std::cout << "CUBLAS ����ʵ��������" << std::endl;
//		}
//		getchar();
//
//	}
//	// ͬ������
//	cudaThreadSynchronize();
//	// ���ݽ�������˺����еĲ��������庬����ο������ֲᡣ
//	float a = 1; float b = 0;
//	// ������ˡ��ú�����Ȼ���������������������
//
//	status = cublasSgemm
//	(
//		//A(primary_hash_proejection)
//		//B(description_Mat)
//		//C(result)
//		handle, //blas�����
//		CUBLAS_OP_N, //����A��ת��
//		CUBLAS_OP_N, //����B��ת��
//		descriptionDimension, //����A��C������,Ҳ�����������
//		mat_I_cols_count, //����B��C��������Ҳ�����������
//		descriptionDimension, //����A����������B������
//		&a,  //alpha��ֵ
//		primary_hash_projection_data_device, //�����A
//		descriptionDimension, //A��leading dimension,����Ϊ����������
//		descriptions_GPU, //�Ҿ��� B
//		descriptionDimension, //����B��leading dimension,����Ϊ����������
//		&b,             //beta��ֵ
//		hash_base_GPU, //�������C
//		descriptionDimension//�������C��leading dimension,����Ϊ����������
//	);
//	cublasDestroy(handle);
//}
//
//extern "C"
//void determine_buket_index_for_each_group
//(
//	//������C
//	float *secondary_projection_GPU,
//	//��ʽA
//	float *secondary_hash_projection_j,
//	//��ʽB
//	const float *descriptions_GPU,
//	//��ʽA������  10
//	int secondary_rows,
//	//��ʽA������ 128
//	int secondary_cols,
//	//��ʽB������ 
//	int descrptions_cols
//	//��ʽB����������descriptionDimension
//)
//{
//	/*dim3 threads(1, 1);
//	dim3 grid(1, 1);*/
//	// ��������ʼ�� CUBLAS �����
//	cublasHandle_t handle;
//	int status = cublasCreate(&handle);
//	if (status != CUBLAS_STATUS_SUCCESS)
//	{
//		if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
//			std::cout << "CUBLAS ����ʵ��������" << std::endl;
//		}
//		getchar();
//
//	}
//	// ͬ������
//	cudaThreadSynchronize();
//	// ���ݽ�������˺����еĲ��������庬����ο������ֲᡣ
//	float a = 1; float b = 0;
//	// ������ˡ��ú�����Ȼ���������������������
//
//	status = cublasSgemm
//	(
//		//A(primary_hash_proejection)
//		//B(description_Mat)
//		//C(result)
//		handle, //blas�����
//		CUBLAS_OP_N, //����A��ת��
//		CUBLAS_OP_N, //����Bת��
//		secondary_rows, //����A��C������,Ҳ�����������
//		descrptions_cols, //����B��C��������Ҳ�����������
//		descriptionDimension, //����A����������B������
//		&a,  //alpha��ֵ
//		secondary_hash_projection_j, //�����A
//		secondary_rows, //A��leading dimension,����Ϊ����������
//		descriptions_GPU, //�Ҿ��� B
//		descriptionDimension, //����B��leading dimension,����Ϊ����������
//		&b,             //beta��ֵ
//		secondary_projection_GPU, //�������C
//		secondary_rows//�������C��leading dimension
//	);
//	cublasDestroy(handle);
//}