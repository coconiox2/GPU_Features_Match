#include "HashDataGeneratorGPU.h"
#include <stdio.h>

__global__ void printFirstProjMatrixGPU_kernel(const cudaPitchedPtr devPitchedPtr)
{
	int idX = blockIdx.x * blockDim.x + threadIdx.x;
	int idY = blockIdx.y * blockDim.y + threadIdx.y;
		
	if(idX < kDimSiftData && idY < kCntBucketBit){
		char* devptr = (char*)devPitchedPtr.ptr;
		size_t pitch = devPitchedPtr.pitch;
		size_t slicepitch = pitch * kCntBucketBit;

		for (int d = 0; d < kCntBucketGroup; d++){
			char *slice = devptr + d * slicepitch;
			int *row = (int*)(slice + idY * pitch);
			printf("%d\n", *(row + idX));
		}
	}
	
	//Standard Code to access 3d data allocated with cudamalloc3d
	//char *devPtr = (char*)devPitchedPtr.ptr;
	//size_t pitch = devPitchedPtr.pitch;
	//size_t slicePitch = pitch * kCntBucketBit;
	//
	//for (int d = 0; d < kCntBucketGroup; d++)
	//{
	//	char* slice = devPtr + d * slicePitch;
	//	for (int y = 0; y < kCntBucketBit; ++y) {
	//		float* row = (float*)(slice + y * pitch);
	//		for (int x = 0; x < kDimSiftData; ++x) {
	//			printf("%f\n", row[x]); //GPU can only print 4096 rows				
	//		}
	//	}
	//}	
}

__global__ void printSecondProjMatrixGPU_kernel(const int *projMat, const size_t pitch)
{
	//int idX = blockIdx.x * blockDim.x + threadIdx.x;
	//int idY = blockIdx.y * blockDim.y + threadIdx.y;
	
	for (int i = 0; i < kDimSiftData; i++) {
		int *matElement = (int*)((char*)projMat + i * pitch);
		for (int j = 0; j < kDimSiftData; j++) {
			printf("%d:  add:%p\n", *(matElement + j), (matElement + j));
		}
	}	
}

void printFirstProjMatrixGPU(const cudaPitchedPtr devPitchedPtr)
{
	dim3 block(32,32);
	dim3 grid((kDimSiftData + 32 - 1) / 32, (kCntBucketBit + 32 - 1) / 32);
	
	printf("GPU print\n");
	printFirstProjMatrixGPU_kernel <<<grid, block>>> (devPitchedPtr);
	//printFirstProjMatrixGPU_kernel <<<1,1>>> (devPitchedPtr);
	cudaDeviceSynchronize();
}

void printSecondProjMatrixGPU(const int *projMat, const size_t pitch)
{
	dim3 block(WARP_SIZE, WARP_SIZE);
	dim3 grid((kDimSiftData + WARP_SIZE - 1) / WARP_SIZE, (kDimSiftData + WARP_SIZE - 1) / WARP_SIZE);

	//printSecondProjMatrixGPU_kernel <<<grid, block >>> (projMat, pitch);
	printSecondProjMatrixGPU_kernel <<<1,1>>> (projMat, pitch);
	cudaDeviceSynchronize();
}

