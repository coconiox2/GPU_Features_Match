#include <iostream>
#include <thrust/sort.h>
//#include <math.h>
#include "HashCalculatorGPU.h"
#include "Common.h"

#define BLOCK_SIZE 32
//#define TOP_PER_THREAD_HAMMING_LIST_SIZE 32 * 10


__global__ void hamming_distance_kernel_revised_old(ImageData *imageData1, ImageData *imageData2, HashData *hashData1, HashData *hashData2, uint16_t* deviceCandidateList, uint8_t* deviceCandidateCntList)
{
	int index = blockIdx.x;

	__shared__ uint16_t bucket_id;
	__shared__ uint16_t totalCandidateSiftPoints;
	__shared__ uint16_t candidateSiftPointList[MAX_CANDIDATE_LIST_SIZE];
	__shared__ uint16_t topCandidateIndex[BLOCK_SIZE * kCntCandidateTopMin];
	__shared__ uint8_t topCandidateHammingDist[BLOCK_SIZE * kCntCandidateTopMin];
	__shared__ uint16_t sortedCandidatePoints[BLOCK_SIZE * kCntCandidateTopMin];
	
	if (index < imageData1->cntPoint) {
		__shared__ int dataIndex;
		__shared__ int countSiftPointsBucket;

		dataIndex = 0;
		totalCandidateSiftPoints = 0;
		countSiftPointsBucket = 0;


		for (int i = 0; i < kCntBucketGroup; i++) {
			if (threadIdx.x == 0) {

				bucket_id = imageData1->deviceBucketIDSiftPoint[imageData1->cntPoint * i + index];
				dataIndex += countSiftPointsBucket;
				countSiftPointsBucket = hashData2->deviceCntSiftPointInBucket[i * kCntBucketPerGroup + bucket_id];
				totalCandidateSiftPoints = totalCandidateSiftPoints + hashData2->deviceCntSiftPointInBucket[i * kCntBucketPerGroup + bucket_id];
			}

			__syncthreads();
			for (int j = 0; j < (countSiftPointsBucket + BLOCK_SIZE - 1) / BLOCK_SIZE; j++) {
				if ((BLOCK_SIZE * j + threadIdx.x) < countSiftPointsBucket) {
					candidateSiftPointList[dataIndex + BLOCK_SIZE * j + threadIdx.x] = hashData2->deviceBucketList[i * kCntBucketPerGroup + bucket_id][BLOCK_SIZE * j + threadIdx.x];
				}
			}
		}

		
		__syncthreads();

		int64_t firstImageHashA, firstImageHashB, secondImageHashA, secondImageHashB;
		firstImageHashA = imageData1->compHashDataPtrList[index * kDimCompHashData];
		firstImageHashB = imageData1->compHashDataPtrList[index * kDimCompHashData + 1];
		int start_index = (threadIdx.x * kCntCandidateTopMin);
		int max_value = kDimSiftData + 1;
		int hammingDistanceTemp;

		//initialize the array
		for (int j = 0; j < kCntCandidateTopMin; j++)
		{
			topCandidateHammingDist[start_index + j] = max_value;
			topCandidateIndex[start_index + j] = imageData1->cntPoint + 1;
		}
		//int end_index = start_index + (kCntCandidateTopMin - 1);
		for (int i = 0; i < (totalCandidateSiftPoints + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
			if ((i * BLOCK_SIZE + threadIdx.x) < totalCandidateSiftPoints) {

				secondImageHashA = imageData2->compHashDataPtrList[candidateSiftPointList[i * BLOCK_SIZE + threadIdx.x] * kDimCompHashData];
				secondImageHashB = imageData2->compHashDataPtrList[candidateSiftPointList[i * BLOCK_SIZE + threadIdx.x] * kDimCompHashData + 1];
				hammingDistanceTemp = __popcll(firstImageHashA ^ secondImageHashA) + __popcll(firstImageHashB ^ secondImageHashB);
				
				
				//do insertion sort
				int to_insert_index = candidateSiftPointList[i * BLOCK_SIZE + threadIdx.x];
				//int to_insert_index = i * BLOCK_SIZE + threadIdx.x;
				bool duplicate = false;
				
				for (int j = 0; j < kCntCandidateTopMin; j++)
				{
					//inserted = false;

					//candidate should be in the list
					if (hammingDistanceTemp < topCandidateHammingDist[start_index + j])
					{
						//inserted = true;

						//look iif there is a duplicate in the array already.
						for (int k = 0; k < kCntCandidateTopMin; k++)
						{
							if (topCandidateIndex[start_index + k] == to_insert_index)
							{
								duplicate = true;
								break;
							}
						}

						//don't proceed if there is a duplicate
						if (duplicate)
						{
							break;
						}

						//move all of the elements up 1 position
						for (int k = kCntCandidateTopMin - 2; k >= j; k--)
						{
							topCandidateHammingDist[start_index + k + 1] = topCandidateHammingDist[k];
							topCandidateIndex[start_index + k + 1] = topCandidateHammingDist[k];
						}
						topCandidateHammingDist[start_index + j] = hammingDistanceTemp;

						topCandidateIndex[start_index + j] = to_insert_index;
						break;
					}
				}
			}
		}

		__syncthreads();


		__shared__ int count;

		count = 0;

		uint8_t my_h = topCandidateHammingDist[start_index];
		uint16_t my_can = topCandidateIndex[start_index];
		int my_count = 0;
		uint8_t min_h = 0;
		while (min_h != (max_value + 1)) {
			min_h = my_h;
			for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
				min_h = min(__shfl_xor(min_h, stride), min_h);
				__syncthreads();
			}
			if (min_h == my_h) {
				if (my_count < kCntCandidateTopMin  && my_h < max_value) {
					int index = atomicAdd(&count, 1);
					sortedCandidatePoints[index] = my_can;
					my_count++;
					if (my_count < kCntCandidateTopMin) {
						start_index++;
						my_h = topCandidateHammingDist[start_index];
						my_can = topCandidateIndex[start_index];
					}
					else
						my_h = max_value + 1;
				}
				else
					my_h = max_value + 1;
			}
		}
		__syncthreads();

	}


	__shared__ uint16_t topKCandidates[kCntCandidateTopMin];

	if (index < imageData1->cntPoint) {
		int candidatesFoundCnt;
		if (threadIdx.x == 0) {

			candidatesFoundCnt = 0;
			uint16_t candidate;
			bool duplicate;

			for (int i = 0; i < (BLOCK_SIZE * kCntCandidateTopMin); i++) {
				duplicate = false;
				candidate = sortedCandidatePoints[i];
				for (int j = 0; j < candidatesFoundCnt; j++) {
					if (candidate == topKCandidates[j])
					{
						duplicate = true;
					}
				}
				if (duplicate == true) {
					continue;
				}
				topKCandidates[candidatesFoundCnt] = candidate;
				candidatesFoundCnt++;
				if (candidatesFoundCnt == kCntCandidateTopMin)
				{
					break;
				}
			}

			for (int i = 0; i < candidatesFoundCnt; i++) {
				deviceCandidateList[index * kCntCandidateTopMin + i] = topKCandidates[i];
			}
			deviceCandidateCntList[index] = candidatesFoundCnt;

		}
	}

}

__global__ void hamming_distance_kernel_revised(ImageData *imageData1, ImageData *imageData2, HashData *hashData1, HashData *hashData2, uint16_t* deviceCandidateList, uint8_t* deviceCandidateCntList, uint16_t* hammingDistanceBuckets)
{
	int index = blockIdx.x;

	__shared__ int dataIndex;
	__shared__ int countSiftPointsBucket;
	__shared__ int block_start_index;
	__shared__ uint16_t bucket_id;
	__shared__ uint16_t totalCandidateSiftPoints;
	__shared__ uint16_t candidateSiftPointList[MAX_CANDIDATE_LIST_SIZE];
	__shared__ int hammingDistanceBucketCounts[maxHammingDistance + 1];
	
	if (index < imageData1->cntPoint) {
		
		block_start_index = index * (maxHammingDistance + 1) * 150;
	
		int currentHammingDistanceBucketIndex;
		
		dataIndex = 0;
		totalCandidateSiftPoints = 0;
		countSiftPointsBucket = 0;


		for (int i = 0; i < kCntBucketGroup; i++) {
			if (threadIdx.x == 0) {

				bucket_id = imageData1->deviceBucketIDSiftPoint[imageData1->cntPoint * i + index];
				dataIndex += countSiftPointsBucket;
				countSiftPointsBucket = hashData2->deviceCntSiftPointInBucket[i * kCntBucketPerGroup + bucket_id];
				totalCandidateSiftPoints = totalCandidateSiftPoints + hashData2->deviceCntSiftPointInBucket[i * kCntBucketPerGroup + bucket_id];
				
			}

			__syncthreads();
			for (int j = 0; j < (countSiftPointsBucket + BLOCK_SIZE - 1) / BLOCK_SIZE; j++) {
				if ((BLOCK_SIZE * j + threadIdx.x) < countSiftPointsBucket) {
					candidateSiftPointList[dataIndex + BLOCK_SIZE * j + threadIdx.x] = hashData2->deviceBucketList[i * kCntBucketPerGroup + bucket_id][BLOCK_SIZE * j + threadIdx.x];
				}
			}
		}

		int initializing_index;
		//initialize hammingDistanceBucketCounts to 0
		for(int i = 0; i < (maxHammingDistance + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE; i++)
		{
			initializing_index = (i * BLOCK_SIZE) + threadIdx.x;
			if(initializing_index <= maxHammingDistance)
			{
				hammingDistanceBucketCounts[initializing_index] = 0;
			}
		}
		
		__syncthreads();

		int64_t firstImageHashA, firstImageHashB, secondImageHashA, secondImageHashB;
		firstImageHashA = imageData1->compHashDataPtrList[index * kDimCompHashData];
		firstImageHashB = imageData1->compHashDataPtrList[index * kDimCompHashData + 1];
		int start_index = (threadIdx.x * kCntCandidateTopMin);
		int max_value = kDimSiftData + 1;
		int hammingDistanceTemp;
		
		//int end_index = start_index + (kCntCandidateTopMin - 1);
		int index_to_write;
		int offset;
		for (int i = 0; i < (totalCandidateSiftPoints + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
			if ((i * BLOCK_SIZE + threadIdx.x) < totalCandidateSiftPoints) {

				secondImageHashA = imageData2->compHashDataPtrList[candidateSiftPointList[i * BLOCK_SIZE + threadIdx.x] * kDimCompHashData];
				secondImageHashB = imageData2->compHashDataPtrList[candidateSiftPointList[i * BLOCK_SIZE + threadIdx.x] * kDimCompHashData + 1];
				hammingDistanceTemp = __popcll(firstImageHashA ^ secondImageHashA) + __popcll(firstImageHashB ^ secondImageHashB);
				
				offset = atomicAdd(&(hammingDistanceBucketCounts[hammingDistanceTemp]), 1);
				
				index_to_write = ((150 ) *hammingDistanceTemp) + offset + block_start_index;
				hammingDistanceBuckets[index_to_write] = candidateSiftPointList[i * BLOCK_SIZE + threadIdx.x];
			}
		}

	
	__syncthreads();

	__shared__ uint16_t topKCandidates[kCntCandidateTopMin];

	if (threadIdx.x == 0 && index < imageData1->cntPoint) {
		
		int candidatesFoundCnt;
		
			candidatesFoundCnt = 0;
			uint16_t candidate;
			bool duplicate;
			int totalcounts = 0;
			for(int i = 0; i < maxHammingDistance + 1; i++)
			{
				
				totalcounts += hammingDistanceBucketCounts[i];
				for(int j = 0; j < hammingDistanceBucketCounts[i]; j++)
				{
					
					duplicate = false;
					candidate = hammingDistanceBuckets[block_start_index + (i * (150)) + j];
					for(int k = 0; k < candidatesFoundCnt; k++)
					 {
						if(candidate == topKCandidates[k])
						{
							duplicate = true;
						}
					}
					
					if(duplicate == true)
					{
						continue;
					}
					
					 topKCandidates[candidatesFoundCnt] = candidate;
					  candidatesFoundCnt++;
					  if(candidatesFoundCnt == kCntCandidateTopMin)
					  {
						  break;
					  }
				 }
				 if(candidatesFoundCnt == kCntCandidateTopMin)
				 {
					 break;
				 }
			
			}
			
			
			deviceCandidateCntList[index] = candidatesFoundCnt;
			for (int w = 0; w < candidatesFoundCnt; w++) 
			{
					deviceCandidateList[index * kCntCandidateTopMin + w] = topKCandidates[w];
			}
			
	}
		
}

}


void compute_hamming_distance_GPU(ImageData *deviceptr1, ImageData* deviceptr2, HashData *hashData1, HashData *hashData2, int siftCount, uint16_t* deviceCandidateList, uint8_t* deviceCandidateCntList, uint16_t* deviceHammingBuckets, cudaStream_t *stream) {

	dim3 block(BLOCK_SIZE);
	dim3 grid(siftCount);
	hamming_distance_kernel_revised <<<grid, block>>>(deviceptr1, deviceptr2, hashData1, hashData2, deviceCandidateList, deviceCandidateCntList, deviceHammingBuckets);
	//hamming_distance_kernel_revised_old <<<grid, block >>>(deviceptr1, deviceptr2, hashData1, hashData2, deviceCandidateList, deviceCandidateCntList);
	CUDA_CHECK_ERROR;
	cudaDeviceSynchronize();
}