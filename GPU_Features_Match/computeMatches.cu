#include <iostream>
//CUDA V 10.2
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#define BLOCK_NUM 32   //块数量
#define THREAD_NUM 256 // 每个块中的线程数
#define R_SIZE BLOCK_NUM * THREAD_NUM
#define M_SIZE R_SIZE * R_SIZE

extern "C" 
int testCUDACPP(int a,int b) {
	int c = a + b;
	return c;
}

__global__ void EUGPU_square(float *desGPU1MinusDesGPU2, float *desGPU1MinusDesGPU2Square)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	desGPU1MinusDesGPU2Square[i] = desGPU1MinusDesGPU2[i] * desGPU1MinusDesGPU2[i];
}

__global__ void EUGPU_minus(float *desGPU1, float *desGPU2, float *desGPU1MinusDesGPU2)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	desGPU1MinusDesGPU2[i] = desGPU1[i] - desGPU2[i];
}

//__global__ void GPU_Mul(float *primary_GPU, float *descriptor_GPU, float *result_GPU)
//{
//	const int bid = blockIdx.x;
//	const int tid = threadIdx.x;
//	// 每个线程计算一行
//	const int row = bid * THREAD_NUM + tid;
//	for (int c = 0; c < R_SIZE; c++) {
//		for (int n = 0; n < R_SIZE; n++) {
//			result_GPU[row*R_SIZE + c] += primary_GPU[row*R_SIZE + n] * descriptor_GPU[n*R_SIZE + c];
//		}
//	}
//}

extern "C"
float computeEuclideanDistance
(
	const unsigned char * descriptionData2,
	const unsigned char * descriptionData1,
	size_t size
)
{
	float * descriptionDataFloat1 = (float *)malloc(sizeof(float) * size);
	float * descriptionDataFloat2 = (float *)malloc(sizeof(float) * size);
	for (int i = 0; i < size; i++) {
		descriptionDataFloat1[i] = static_cast<float>(descriptionData1[i]);
		descriptionDataFloat2[i] = static_cast<float>(descriptionData2[i]);
	}

	float *desGPU1;
	float *desGPU2;

	cudaMalloc((void**)&desGPU1, sizeof(float) * size);
	cudaMalloc((void**)&desGPU2, sizeof(float) * size);

	cudaMemcpy(desGPU1, descriptionDataFloat1, sizeof(float) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(desGPU2, descriptionDataFloat2, sizeof(float) * size, cudaMemcpyHostToDevice);

	/*dim3 dimBlock(size);
	dim3 dimGrid(1);*/
	dim3 dimBlock(1024);
	dim3 dimGrid(256);

	float *resultGPU;
	cudaMalloc((void**)&resultGPU, sizeof(float));

	float *desGPU1MinusDesGPU2;
	cudaMalloc((void **)&desGPU1MinusDesGPU2, sizeof(float) * size);

	float *desGPU1MinusDesGPU2Square;
	cudaMalloc((void **)&desGPU1MinusDesGPU2Square, sizeof(float) * size);

	//执行kernel
	cudaThreadSynchronize();
	EUGPU_minus << <dimGrid, dimBlock >> > (desGPU1, desGPU2, desGPU1MinusDesGPU2);

	cudaThreadSynchronize();
	EUGPU_square << <dimGrid, dimBlock >> > (desGPU1MinusDesGPU2, desGPU1MinusDesGPU2Square);

	float *desGPU1MinusDesGPU2SquareCPU = (float *)malloc(sizeof(float) * size);
	cudaMemcpy(desGPU1MinusDesGPU2SquareCPU, desGPU1MinusDesGPU2Square, sizeof(float) * size, cudaMemcpyDeviceToHost);

	float result = 0.0;
	for (int i = 0; i < size; i++) {
		result += desGPU1MinusDesGPU2SquareCPU[i];
	}
	free(desGPU1MinusDesGPU2SquareCPU);

	cudaFree(resultGPU);
	cudaFree(desGPU1MinusDesGPU2);
	cudaFree(desGPU1MinusDesGPU2Square);

	cudaFree(desGPU1);
	cudaFree(desGPU2);

	free(descriptionDataFloat1);
	free(descriptionDataFloat2);

	return result;
}

//extern "C"
//float primaryMulDescriptor
//(
//	float * descriptor,
//	float * result,
//	size_t size
//) 
//{
//	float *descriptor_GPU;
//	cudaMalloc((void**)&descriptor_GPU, sizeof(float) * size);
//	cudaMemcpy(descriptor_GPU, descriptor, sizeof(float) * size, cudaMemcpyHostToDevice);
//	
//	float *result_GPU;
//	cudaMalloc((void**)&result_GPU, sizeof(float) * size);
//	//cudaMemcpy(desGPU2, descriptionDataFloat2, sizeof(float) * size, cudaMemcpyHostToDevice);
//
//	dim3 dimBlock(1024);
//	dim3 dimGrid(256);
//
//	cudaThreadSynchronize();
//	EUGPU_square << <dimGrid, dimBlock >> > (desGPU1MinusDesGPU2, desGPU1MinusDesGPU2Square);
//}