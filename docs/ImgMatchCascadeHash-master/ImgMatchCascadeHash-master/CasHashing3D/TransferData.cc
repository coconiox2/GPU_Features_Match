#include "TransferData.h"
#include "Common.h"
#include  <iostream>
using namespace std;

void uploadImageDataToDevice(ImageData *h_imgData, int imageNumber)
{
	//transferring the sift feature points
	cudaMalloc(&(h_imgData->deviceSiftDataPtrList),h_imgData->cntPoint * sizeof(int) * kDimSiftData);
	CUDA_CHECK_ERROR;
		
	for (int j = 0; j < h_imgData->cntPoint; j++) {
		
		int start_index = j * kDimSiftData;

		cudaMemcpy(&(h_imgData->deviceSiftDataPtrList[start_index]), h_imgData->siftDataPtrList[j], sizeof(int) * kDimSiftData, cudaMemcpyHostToDevice);
	}
	
	//mallocing bucketIdlist
	cudaMalloc(&(h_imgData->deviceBucketIDSiftPoint), h_imgData->cntPoint * sizeof(uint16_t) * kCntBucketGroup);
	
	//mallocing hashDataPtrList
	cudaMalloc(&(h_imgData->deviceHashDataPtrList),h_imgData->cntPoint * sizeof(uint8_t) * kDimSiftData);

	cudaMalloc(&(h_imgData->compHashDataPtrList), h_imgData->cntPoint * sizeof(uint64_t*) * kDimCompHashData);
	
}

void uploadImagesToDevice(ImageData **d_imgData, ImageData *h_imgData){
	cudaMalloc(d_imgData, sizeof(ImageData) ); 
	
	cudaMemcpy(*d_imgData, h_imgData, sizeof(ImageData), cudaMemcpyHostToDevice);
	
}

uint16_t* mallocCandidatePointsArray(int cntPoint){
	uint16_t* arrayPointer;
	cudaMalloc(&arrayPointer, sizeof(uint16_t) * kCntCandidateTopMin * cntPoint);
	return arrayPointer;
}


uint8_t* mallocCandidateCntList(int cntPoint){
	uint8_t* arrayPointer;
	cudaMalloc(&arrayPointer, sizeof(uint8_t) * cntPoint);
	return arrayPointer;
}

void freeCandidatePointsArray(uint16_t* arrayPointer){
	cudaFree(arrayPointer);
}

void freeCandidateCntList(uint8_t* arrayPointer){
	cudaFree(arrayPointer);
}

void downloadHashData(ImageData *imgData){
	imgData->bucketIDSiftPoint = (uint16_t*)malloc(imgData->cntPoint * sizeof(uint16_t) *kCntBucketGroup);
	cudaMemcpy(imgData->bucketIDSiftPoint, imgData->deviceBucketIDSiftPoint, imgData->cntPoint * sizeof(uint16_t) *kCntBucketGroup, cudaMemcpyDeviceToHost);
}

HashData* uploadBucket(ImageData *imgData){
	HashData *hashData = (HashData*)malloc(sizeof(HashData));
	
	int index = 0;
	int group_index = 0;
	int bucket_id = 0;
	
	//Copy the cnts matrix
	cudaMalloc(&(hashData->deviceCntSiftPointInBucket), sizeof(int) * kCntBucketGroup * kCntBucketPerGroup);
	cudaMemcpy(hashData->deviceCntSiftPointInBucket, imgData->cntSiftPointInBucket, sizeof(int) * kCntBucketGroup * kCntBucketPerGroup, cudaMemcpyHostToDevice);
	
	//Copy the buckets
	BucketElePtr* tempPtrArray = (BucketElePtr*)malloc(sizeof(BucketElePtr*) * kCntBucketGroup * kCntBucketPerGroup);
	BucketElePtr ptr; 
	int count;
	for(int bucketGroup = 0; bucketGroup < kCntBucketGroup; bucketGroup++)
	{
		for(int bucketID = 0; bucketID < kCntBucketPerGroup; bucketID++)
		{
			count = imgData->cntSiftPointInBucket[bucketGroup][bucketID];
			//printf("Count is %d\n", count);
			cudaMalloc(&ptr, sizeof(int) * count);
			cudaMemcpy(ptr, imgData->bucketList[bucketGroup][bucketID], sizeof(int) * count, cudaMemcpyHostToDevice);
			
			tempPtrArray[bucketGroup * kCntBucketPerGroup + bucketID] = ptr;
		}
	}
	cudaMalloc(&(hashData->deviceBucketList), sizeof(BucketElePtr) * kCntBucketGroup * kCntBucketPerGroup);
	cudaMemcpy(hashData->deviceBucketList, tempPtrArray, sizeof(BucketElePtr) * kCntBucketGroup * kCntBucketPerGroup, cudaMemcpyHostToDevice);
	
	//Copy the hashData struct
	HashData *deviceHashData;
	cudaMalloc(&deviceHashData, sizeof(HashData));
	cudaMemcpy(deviceHashData, hashData, sizeof(HashData), cudaMemcpyHostToDevice);
	return deviceHashData;
}

void freeSiftData(ImageData *imgData){
	cudaFree(imgData->deviceSiftDataPtrList);
}
