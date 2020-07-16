#include <stdio.h>
#include "BucketBuilderGPU.h"

__global__ void count_element_bucket_kernel(ImageData *imageData, int *cntEleInBucketDevice, int bucketGroupID, int imageIndex) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int bucketIndex = 0;
	
	if (index < imageData->cntPoint) {
		
		bucketIndex = imageData->deviceBucketIDSiftPoint[bucketGroupID * imageData->cntPoint + index];
		atomicAdd(&(cntEleInBucketDevice[bucketIndex]), 1);						
		
		__syncthreads();
		
	}
}


void count_element_bucket_GPU(ImageData *imageData, int *cntEleInBucketDevice, int siftCount, int bucketGroupID, int imageIndex) {

	dim3 block(1024);
	dim3 grid((siftCount + block.x - 1) / block.x);

	//printf("sift point count: %d & sift = %d\n", grid.x, siftCount);
	count_element_bucket_kernel <<<grid, block >>> (imageData, cntEleInBucketDevice, bucketGroupID, imageIndex);
	
}