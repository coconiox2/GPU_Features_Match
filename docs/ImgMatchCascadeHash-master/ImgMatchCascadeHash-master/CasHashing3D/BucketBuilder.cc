//#include <cstring>
#include <cstdlib>
#include <iostream>
#include "BucketBuilder.h"

void BucketBuilder::Build_Revised(ImageData **deviceImageData, ImageData& hostImageData, int siftCount, int imageIndex)
{
	int cntEleInBucket[kCntBucketPerGroup]; // accumulator; the number of SIFT points in each bucket
	int *cntEleInBucketDevice;

	cudaMalloc(&cntEleInBucketDevice, sizeof(int) * kCntBucketPerGroup);

	for (int groupIndex = 0; groupIndex < kCntBucketGroup; groupIndex++) {
		
		cudaMemset(cntEleInBucketDevice, 0, sizeof(int) * kCntBucketPerGroup);
		count_element_bucket_GPU(*deviceImageData, cntEleInBucketDevice, siftCount, groupIndex, imageIndex);		
		cudaMemcpy(cntEleInBucket, cntEleInBucketDevice, sizeof(int) * kCntBucketPerGroup, cudaMemcpyDeviceToHost);

	
		// allocate space for <imageData.bucketList>
		for (int bucketID = 0; bucketID < kCntBucketPerGroup; bucketID++) {
			hostImageData.cntSiftPointInBucket[groupIndex][bucketID] = cntEleInBucket[bucketID];
			hostImageData.bucketList[groupIndex][bucketID] = (int*)malloc(sizeof(int) * cntEleInBucket[bucketID]);
			cntEleInBucket[bucketID] = 0;
		}

		//build_bucket_GPU();
		//// assign the index of each SIFT point to <imageData.bucketList>
		//for (int dataIndex = 0; dataIndex < hostImageData.cntPoint; dataIndex++) {
		//	int bucketID = hostImageData.bucketIDSiftPoint[groupIndex * hostImageData.cntPoint + dataIndex];
		//	hostImageData.bucketList[groupIndex][bucketID][cntEleInBucket[bucketID]++] = dataIndex;
		//	//printf("sseidfsd = %d", hostImageData.bucketList[groupIndex][bucketID][cntEleInBucket[bucketID]]);
		//}
	}


	//int cntEleInBucket[kCntBucketPerGroup]; // accumulator; the number of SIFT points in each bucket
	//int *cntEleInBucketDevice;

	//cudaMalloc(&cntEleInBucketDevice, sizeof(int) * kCntBucketPerGroup);
	//
	////printf("address:%p", cntEleInBucketDevice);

	//for (int groupIndex = 0; groupIndex < kCntBucketGroup; groupIndex++) {

	//	cudaMemset(cntEleInBucketDevice, 0, sizeof(int) * kCntBucketPerGroup);
	//	count_element_bucket_GPU(*deviceImageData, cntEleInBucketDevice, siftCount, groupIndex, imageIndex);		
	//	cudaMemcpy(cntEleInBucket, cntEleInBucketDevice, sizeof(int) * kCntBucketPerGroup, cudaMemcpyDeviceToHost);
	//			
	//	// allocate space for <imageData.bucketList>
	//	for (int bucketid = 0; bucketid < kCntBucketGroup; bucketid++) {
	//		hostImageData.cntSiftPointInBucket[groupIndex][bucketid] = cntEleInBucket[bucketid];
	//		hostImageData.bucketList[groupIndex][bucketid] = (int*)malloc(sizeof(int) * cntEleInBucket[bucketid]);
	//		cntEleInBucket[bucketid] = 0;
	//	}

	//	//// assign the index of each SIFT point to <imageData.bucketList>
	//	for (int dataIndex = 0; dataIndex < hostImageData.cntPoint; dataIndex++) {
	//		int bucketID = hostImageData.bucketIDSiftPoint[groupIndex * hostImageData.cntPoint + dataIndex];
	//		/*if (groupIndex == 0)
	//			printf("bucket id:%d\n", bucketID);*/
	//		//hostImageData.bucketList[groupIndex][bucketID][cntEleInBucket[bucketID]++] = dataIndex;
	//		if (groupIndex == 0 && dataIndex == 0)
	//			printf("val add = %p\n", hostImageData.bucketList[groupIndex][bucketID][0]);
	//	}
	//			
	//}

}

void BucketBuilder::Build(ImageData& imageData, int imageIndex)
{
	static int cntEleInBucket[kCntBucketPerGroup]; // accumulator; the number of SIFT points in each bucket
	int *cntEleInBucketDevice;

	cudaMalloc(&cntEleInBucketDevice, sizeof(int) * kCntBucketPerGroup);

	for (int groupIndex = 0; groupIndex < kCntBucketGroup; groupIndex++) {
		// initialize <cntEleInBucket>
		for (int bucketID = 0; bucketID < kCntBucketPerGroup; bucketID++) {
			cntEleInBucket[bucketID] = 0;
		}
		//memset(cntEleInBucket, 0, sizeof(int) * kCntBucketPerGroup);

		// count the number of SIFT points falling into each bucket
		for (int dataIndex = 0; dataIndex < imageData.cntPoint; dataIndex++) {			
			cntEleInBucket[imageData.bucketIDSiftPoint[groupIndex * imageData.cntPoint + dataIndex]]++;
		}
		/*
		printf("serial");
		for (int i = 0; i < 256; i++) {
			if (imageIndex == 0)
				printf("%d=%d\n", groupIndex, cntEleInBucket[i]);
		}
*/
		// allocate space for <imageData.bucketList>
		for (int bucketID = 0; bucketID < kCntBucketPerGroup; bucketID++) {
			imageData.cntSiftPointInBucket[groupIndex][bucketID] = cntEleInBucket[bucketID];
			imageData.bucketList[groupIndex][bucketID] = (int*)malloc(sizeof(int) * cntEleInBucket[bucketID]);
			cntEleInBucket[bucketID] = 0;
		}

		// assign the index of each SIFT point to <imageData.bucketList>
		for (int dataIndex = 0; dataIndex < imageData.cntPoint; dataIndex++) {
			int bucketID = imageData.bucketIDSiftPoint[groupIndex * imageData.cntPoint + dataIndex];
			imageData.bucketList[groupIndex][bucketID][cntEleInBucket[bucketID]++] = dataIndex;
			//printf("sseidfsd = %d", imageData.bucketList[groupIndex][bucketID][cntEleInBucket[bucketID]]);
		}
	}
}

