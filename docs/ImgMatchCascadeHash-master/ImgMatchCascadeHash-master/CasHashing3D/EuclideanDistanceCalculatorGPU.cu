#include <stdio.h>
#include "EuclideanDistanceCalculatorGPU.h"
#include "ImageMatchPair.h"

__global__ void calculate_euclidean_distance_kernel(ImageData *imageData1, ImageData *imageData2, uint16_t *deviceCandidateList, uint8_t *deviceCandidateCntList,int  *deviceMatchingPoints, int imageIndex){

	int index = blockIdx.x;
	int tid = threadIdx.x;

	if (index < imageData1->cntPoint) {
				
		double candidateDistListTop[kCntCandidateTopMin];
		int cntCandidateFound = deviceCandidateCntList[index];
		
		for (int candidateIndex = 0; candidateIndex < cntCandidateFound; candidateIndex++) {
			
			__shared__ double distEuclid[kDimSiftData];
			__shared__ double diff[kDimSiftData];
			
			distEuclid[tid] = 0.0f;
			
			int dataIndex_2 = deviceCandidateList[index * kCntCandidateTopMin + candidateIndex];
			
			diff[tid] = imageData1->deviceSiftDataPtrList[index * kDimSiftData + tid] - imageData2->deviceSiftDataPtrList[dataIndex_2 * kDimSiftData + tid];
			
			__syncthreads();
			distEuclid[tid] = diff[tid] * diff[tid];
			__syncthreads();

			for (int stride = kDimSiftData / 2; stride > 0; stride >>= 1) {
				if (tid < stride) {
					distEuclid[tid] += distEuclid[tid + stride];					
				}
				__syncthreads();
			}
			if (tid == 0) {
				candidateDistListTop[candidateIndex] = distEuclid[tid];				
			}			
		}
				
		if (tid == 0) {
			deviceMatchingPoints[index] = findMinValIndex_device(index, cntCandidateFound, deviceCandidateList, &candidateDistListTop[0]);		
		}
	}
}

void compute_euclidean_distance_GPU(ImageData *imageData1, ImageData *imageData2, int siftCount, uint16_t *deviceCandidateList, uint8_t *deviceCandidateCntList, int *deviceMatchingPoints,int imageIndex, cudaStream_t *stream) {

	dim3 block(kDimSiftData);
	dim3 grid((siftCount * block.x + block.x - 1) / block.x);
	
	calculate_euclidean_distance_kernel<<<grid, block, 0>>>(imageData1, imageData2, deviceCandidateList, deviceCandidateCntList, deviceMatchingPoints, imageIndex);
	//cudaDeviceSynchronize();
	
}

__device__ int findMinValIndex_device(int data_index, int cntCandidateFound, uint16_t* hostCandidateList, double* candidateDistListTop) {

	double minVal_1 = 0.0;
	int minValInd_1 = -1;
	double minVal_2 = 0.0;
	int minValInd_2 = -1;

	for (int candidateIndex = 0; candidateIndex < cntCandidateFound; candidateIndex++) {
		if (minValInd_2 == -1 || minVal_2 > candidateDistListTop[candidateIndex]) {
			minVal_2 = candidateDistListTop[candidateIndex];
			minValInd_2 = hostCandidateList[data_index * kCntCandidateTopMin + candidateIndex];
		}
		if (minValInd_1 == -1 || minVal_1 > minVal_2) {
			minVal_1 = minVal_1 + minVal_2;
			minVal_2 = minVal_1 - minVal_2;
			minVal_1 = minVal_1 - minVal_2;
			minValInd_1 = minValInd_1 + minValInd_2;
			minValInd_2 = minValInd_1 - minValInd_2;
			minValInd_1 = minValInd_1 - minValInd_2;
		}
	}

	if (minVal_1 < minVal_2 * matchThreshold) {
		return minValInd_1;
	}
	else
		return -1;
}
