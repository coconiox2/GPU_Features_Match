#include <iostream>
//CUDA V 10.2
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

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

	//Ö´ÐÐkernel
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

