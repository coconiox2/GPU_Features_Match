#include <iostream>
#include <thrust/sort.h>
//#include <inttypes.h>
#include "kernel_test.h"
#include "Common.h"

#define BLOCK_SIZE 32
#define TOP_PER_THREAD_HAMMING_LIST_SIZE 32 * 10

__global__ void compute_hash_kernel(ImageData *ptr, float* d_firstProjMat, float *d_secondProjMat, int imageIndex)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < ptr->cntPoint) {
		
		//if (index == 2) {		
		float sumFirstHash;			
		for (int m = 0; m < kCntBucketGroup; m++){
			int bucketid = 0;
			for (int j = 0; j < kCntBucketBit; j++) {
				sumFirstHash = 0.0f;
				for (int k = 0; k < kDimSiftData; k++) {
					sumFirstHash += ptr->deviceSiftDataPtrList[index * kDimSiftData + k] * d_firstProjMat[m * kDimSiftData * kCntBucketBit + j * kDimSiftData + k];
				}
				/*if (imageIndex == 0 && index == 0 && m == 0)
						printf("sum = %f\n", sumFirstHash);*/
				bucketid = (bucketid << 1) + (sumFirstHash > 0 ? 1 : 0);
				//imgdata.point_to_bucket_map[point_number] = bucketid;
			}
			ptr->deviceBucketIDSiftPoint[ptr->cntPoint * m + index] = bucketid;
			/*if (imageIndex == 0 && index < 35 && m == 0){
				printf("***my index is %d, my image index is %d, the bucketgroup is %d, the bucketId is %d\n", index, imageIndex, m, bucketid);
			}*/
		}

		int sumSecondHash;
		for (int m = 0; m < kDimSiftData; m++) {
			sumSecondHash = 0.0f;
			for (int j = 0; j < kDimSiftData; j++) {
				//if (imageIndex == 0 && index == 0 && m == 0)
				//	printf("sift data kernel= %d, second proj mat kernel= %f\n", ptr->deviceSiftDataPtrList[index * kDimSiftData + j], d_secondProjMat[m * kDimSiftData + j]);
				sumSecondHash += ptr->deviceSiftDataPtrList[index * kDimSiftData + j] * d_secondProjMat[m * kDimSiftData + j];
			}
			ptr->deviceHashDataPtrList[index * kDimSiftData + m] = (sumSecondHash > 0 ? 1 : 0);

			/*if (imageIndex == 0 && (index == 5) && m == 0)
				printf("second sum = %d\n", sumSecondHash);*/
		}
		//printf("second hash value  = %d", ptr->hashDataPtrList[127]);

		// calculate the CompHash code
		// compress <kBitInCompHash> Hash code bits within a single <uint64_t> variable
		
		for (int dimCompHashIndex = 0; dimCompHashIndex < kDimCompHashData; dimCompHashIndex++)
		{
			uint64_t compHashBitVal = 0;
			int dimHashIndexLBound = dimCompHashIndex * kBitInCompHash;
			int dimHashIndexUBound = (dimCompHashIndex + 1) * kBitInCompHash;
			for (int dimHashIndex = dimHashIndexLBound; dimHashIndex < dimHashIndexUBound; dimHashIndex++)
			{				
				compHashBitVal = (compHashBitVal << 1) + ptr->deviceHashDataPtrList[index * kDimSiftData + dimHashIndex]; // set the corresponding bit to 1/0
				
			}
			/*if (imageIndex == 0 && index == 0)
			{
				printf("hashdata %llu\n", compHashBitVal);
			}*/
			ptr->compHashDataPtrList[index * kDimCompHashData + dimCompHashIndex] = compHashBitVal;
			//compHashDataPtr[dimCompHashIndex] = compHashBitVal;
		}

		//
		//if(imageIndex == 1 && index == 2414) {
		//	printf("generated hash value %llu ====%llu\n", ptr->compHashDataPtrList[0], ptr->compHashDataPtrList[1]);
		//	//printf("compress 1: %lld, compress2: %lld\n", ptr->compHashDataPtrList[0], ptr->compHashDataPtrList[1]);
		//}			
	}
}
 
void compute_hash_GPU(ImageData **device_ptr, ImageData *host_ptr ,int img_cnt, float *d_firstProjMat, float *d_secondProjMat, float *firstProjMat)
{
    cudaStream_t streams[img_cnt];
    
    for (int i = 0; i < img_cnt; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
	for(int i = 0; i < img_cnt; i++){
		
		//Parallel
		
		dim3 block(1024);
		dim3 grid((host_ptr->cntPoint + block.x - 1) / block.x);

		//std::cout << "block = " << block.x << " grid = " << grid.x << std::endl;
		compute_hash_kernel<<<grid, block, 0, streams[i]>>>(device_ptr[i], d_firstProjMat, d_secondProjMat, i);
				
		//hello<<<grid, block>>>(device_ptr[i], d_firstProjMat, d_secondProjMat, i);
		
		//CUDA_CHECK_ERROR;
		host_ptr++;
	}
	cudaDeviceSynchronize();
}
/*
__global__ void hamming_distance_kernel(ImageData *imageData1, ImageData *imageData2, HashData *hashData1, HashData *hashData2, uint16_t* deviceCandidateList, uint8_t* deviceCandidateCntList)
{
	//int index = blockIdx.x * blockDim.x;
	int index = blockIdx.x;
	
	//printf("blockidx.x = %d, blockdim.x = %d\n", blockIdx.x, blockDim.x);
	
	int bucket_id;
	__shared__ uint16_t totalCandidateSiftPoints;
	__shared__ uint16_t *candidateSiftPointList;
	__shared__ uint8_t *candidateHammingDistance;
	
	if (index < imageData1->cntPoint) {
		if (threadIdx.x == 0) {

			totalCandidateSiftPoints = 0;
			//printf("%d\n", index);
			//printf("sift count 1: %d, sift count 2: %d \n", imageData1->cntPoint, imageData2->cntPoint);
			//printf("bucket count: %d, bucket point: %d \n", hashData1->deviceCntSiftPointInBucket[0], hashData2->deviceCntSiftPointInBucket[0]);
			
			for (int i = 0; i < kCntBucketGroup; i++)
			{
				bucket_id = imageData1->deviceBucketIDSiftPoint[imageData1->cntPoint * i + index];
				totalCandidateSiftPoints = totalCandidateSiftPoints + hashData2->deviceCntSiftPointInBucket[i * kCntBucketPerGroup + bucket_id];
			}
	
			candidateSiftPointList = new uint16_t[totalCandidateSiftPoints];
			candidateHammingDistance = new uint8_t[totalCandidateSiftPoints];
			int dataIndex = 0;
			
			for (int i = 0; i < kCntBucketGroup; i++) {
				int countSiftPointsBucket = 0;
				bucket_id = imageData1->deviceBucketIDSiftPoint[imageData1->cntPoint * i + index];
				countSiftPointsBucket = hashData2->deviceCntSiftPointInBucket[i * kCntBucketPerGroup + bucket_id];
				
				for (int j = 0; j < countSiftPointsBucket; j++) {
					candidateSiftPointList[dataIndex] = hashData2->deviceBucketList[i * kCntBucketPerGroup + bucket_id][j];
					dataIndex++;
				}
			}			
		}
	}
	
	
	__syncthreads();

	int64_t firstImageHashA, firstImageHashB, secondImageHashA, secondImageHashB;
	firstImageHashA = imageData1->compHashDataPtrList[index * kDimCompHashData];
	firstImageHashB = imageData1->compHashDataPtrList[index * kDimCompHashData + 1];
	
	for (int i = 0; i < int8_t((totalCandidateSiftPoints + BLOCK_SIZE - 1) / BLOCK_SIZE); i++){
		if ((i * BLOCK_SIZE + threadIdx.x) < totalCandidateSiftPoints) {
			secondImageHashA = imageData2->compHashDataPtrList[candidateSiftPointList[i * BLOCK_SIZE + threadIdx.x] * kDimCompHashData];
			secondImageHashB = imageData2->compHashDataPtrList[candidateSiftPointList[i * BLOCK_SIZE + threadIdx.x] * kDimCompHashData + 1];
			candidateHammingDistance[i * BLOCK_SIZE + threadIdx.x] = __popcll(firstImageHashA ^ secondImageHashA) + __popcll(firstImageHashB ^ secondImageHashB);
			//if(index == 0){
					//printf("hamming distance = %d \n", candidateHammingDistance[i * BLOCK_SIZE + threadIdx.x]);
			//}
			
		}
	}
	__syncthreads();
	
	
	__shared__ uint16_t topKCandidates[kCntCandidateTopMin];
	if (index < imageData1->cntPoint) {
		//if (threadIdx.x == 0 && index == 0) {
		int candidatesFoundCnt;
		if(threadIdx.x == 0){
			
			//printf("image no= %d\n", imageData1->cntPoint);
			//printf("%d, %d\n", index, totalCandidateSiftPoints);

			thrust::sort_by_key(thrust::seq, candidateHammingDistance, candidateHammingDistance + totalCandidateSiftPoints, candidateSiftPointList);
			candidatesFoundCnt = 0;
			uint16_t candidate;
			bool duplicate;
			
			for(int i = 0; i < totalCandidateSiftPoints; i++){
				duplicate = false;
				//printf("Index: %d, The sorted hamming distance:%d\n", i, candidateHammingDistance[i]);
				//printf("The corresponding point is:%d\n", candidateSiftPointList[i]);
			
				candidate = candidateSiftPointList[i];
				for(int j = 0; j < candidatesFoundCnt; j++){
					if(candidate == topKCandidates[j])
					{
						duplicate = true;
					}
				}
				if(duplicate == true){
					continue;
				}
				topKCandidates[candidatesFoundCnt] = candidate;
				candidatesFoundCnt++;
				if(candidatesFoundCnt == kCntCandidateTopMin)
				{
					break;
				}
			}
			//printf("max = %d, \n", max);
			
			for(int i = 0; i < candidatesFoundCnt; i++){
				//if (index == 100)
					//printf("index: %d, The candidate is %d\n", index, topKCandidates[i]);
				deviceCandidateList[index * kCntCandidateTopMin + i] = topKCandidates[i];
			}
			deviceCandidateCntList[index] = candidatesFoundCnt;
		}
	}
	
	
	if(threadIdx.x == 0 && index < imageData1->cntPoint){
		delete[] candidateSiftPointList;	
	}
	
}

void compute_hamming_distance_GPU(ImageData *deviceptr1, ImageData* deviceptr2, HashData *hashData1, HashData *hashData2, int siftCount, uint16_t* deviceCandidateList, uint8_t* deviceCandidateCntList){
	
	dim3 block(BLOCK_SIZE);
	dim3 grid(siftCount);
	
	hamming_distance_kernel<<<grid, block>>>(deviceptr1, deviceptr2, hashData1, hashData2, deviceCandidateList, deviceCandidateCntList);
	
	//hamming_distance_kernel<<<grid, block>>>(deviceptr[4], deviceptr[5], hashData1, hashData2, deviceCandidateList, deviceCandidateCntList);
	//cudaDeviceSynchronize();
}
*/
void compute_hashes_serial(ImageData **device_ptr, ImageData *host_ptr ,int img_cnt, float *d_firstProjMat, float *d_secondProjMat, float *firstProjMat){
	
	//Serial	
				
		//for (int dataIndex = 0; dataIndex < host_ptr->cntPoint; dataIndex++)
		//{
		//	// obtain pointers for SIFT feature vector, Hash code and CompHash code
		//	SiftDataPtr siftDataPtr = host_ptr->siftDataPtrList[dataIndex];
		//	int bucketID = 0;
		//	// determine the bucket index for each bucket group

		//	for (int m = 0; m < kCntBucketGroup; m++) {
		//		bucketID = 0;
		//		for (int j = 0; j < kCntBucketBit; j++) {
		//			float sum = 0.0f;
		//			for (int k = 0; k < kDimSiftData; k++) {
		//				if (i == 0 && dataIndex == 0 && m == 0) {
		//					//std::cout << "proj data inside fun" << firstProjMat[m * kDimSiftData * kCntBucketBit + j * kDimSiftData + k] << std::endl;
		//					//printf("sift data = %d, proj data = %f\n", siftDataPtr[k], firstProjMat[m * kDimSiftData * kCntBucketBit + j * kDimSiftData + k]);
		//				}


		//				if (siftDataPtr[k] != 0) {
		//					sum += siftDataPtr[k] * firstProjMat[m * kDimSiftData * kCntBucketBit + j * kDimSiftData + k];
		//				}
		//			}
		//			
		//			if (i == 0 && dataIndex == 0 && m == 0) {
		//				printf("sum = %f\n", sum);
		//			}
		//			bucketID = (bucketID << 1) + (sum > 0 ? 1 : 0);
		//		}

		//		if (i == 0  && dataIndex == 0 && m == 0) {
		//			printf("===============index = %d, group index = %d, bucket id = %d, image no = %d\n===================", dataIndex, m, bucketID, i);
		//		}
		//	}

		//}

		//

	
}


//
//typedef struct StructA {
//	SiftDataPtr *arrval;
//	int flag;
//} StructA;
//
//__global__ void kernel(StructA *in) {
//		
//	for (int i = 0; i < 2; i++) {
//		//printf("flag val = %d\n", in[i].flag);
//		//printf("address = %p\n", in[i].arrval);
//		for (int j = 0; j < 3; j++) {
//			for (int k = 0; k < 5; k++) {
//				//printf("address = %p\n", &in[i].arrval[j * 5 + k]);
//				//printf("val = %d\n", in[i].arrval[j * 5 + k]);
//				printf("val = %d\n", d_arrval[i][j * 5 + k]);
//			}
//		}
//		printf("-------------------------------------------\n");
//	}
//}
//
//void call_kernel_test2()
//{
//	StructA *h_a, *d_a;
//	
//	printf("h_a address:%p\n", h_a);
//	h_a = (StructA*)malloc(2 * sizeof(StructA));
//	printf("h_a address:%p\n", h_a);
//
//	for (int i = 0; i < 2; i++) {
//		printf("h_a[%d] arrval address:%p\n", i, h_a[i].arrval);
//		h_a[i].arrval = (SiftDataPtr*)malloc(3 * sizeof(SiftDataPtr));
//		printf("h_a[%d] arrval address:%p\n", i, h_a[i].arrval);
//		for (int j = 0; j < 3; j++) {
//			h_a[i].arrval[j] = (int*)malloc(5 * sizeof(int));
//		}
//	}
//		
//	int counter = 0;
//	for (int i = 0; i < 2; i++) {
//		for (int j = 0; j < 3; j++) {
//			for (int k = 0; k < 5; k++) {
//				counter += 1;
//				h_a[i].arrval[j][k] = counter;
//			}
//			//printf("\n");
//		}
//		h_a[i].flag = counter;
//	}
//	
//	/*for (int i = 0; i < 2; i++) {
//		for (int j = 0; j < 8; j++) {
//			for (int k = 0; k < 5; k++)
//				printf("val = %d\t", h_a[i].arrval[j][k]);
//			printf("\n");
//		}
//		printf("-------------------------------------------\n");
//	}*/
//		
//	// 1. Allocate device array.
//	
//	size_t sz = 0;
//	cudaDeviceGetLimit(&sz, cudaLimitMallocHeapSize);
//	printf("heap size:%ld\n", sz);
//		
//	cudaMalloc(&d_a, sizeof(StructA) * 2);
//	CUDA_CHECK_ERROR;
//	cudaMemcpy(d_a, h_a, sizeof(StructA) * 2, cudaMemcpyHostToDevice);
//	CUDA_CHECK_ERROR;
//	
//	int *d_arrval[2];
//	
//	for (int i = 0; i < 2; i++) {
//		cudaMalloc(&d_arrval[i], 3 * 5 * sizeof(int));
//		cudaMemcpy(&d_a[i].arrval, &d_arrval[i], sizeof(int*), cudaMemcpyHostToDevice);
//		for (int j = 0; j < 3; j++) {			
//			cudaMemcpy(&d_arrval[i][j], h_a[i].arrval[j], 5 * sizeof(int), cudaMemcpyHostToDevice);
//		}
//	}
//	
//	// 4. Call kernel with host struct as argument
//	kernel<<<1, 1>>>(d_a);
//
//	// 5. Copy pointer from device to host.
//	//cudaMemcpy(h_arr, d_arr, sizeof(int)*10, cudaMemcpyDeviceToHost);
//
//	// 6. Point to host pointer in host struct 
//	//    (or do something else with it if this is not needed)
//	//h_a.arr = h_arr;
//	
//}
//
//class Particle
//{
//public:
//	double *_w;
//};
//
//__global__ void test(Particle *p) {
//
//	int idx = threadIdx.x + blockDim.x*blockIdx.x;
//
//	if (idx == 2) {
//		printf("dev_p[2]._w[2] = %f\n", p[idx]._w[2]);
//	}
//}
//
//void call_kernel_test3()
//{
//	int nParticles = 100;
//	Particle *dev_p;
//	double *w[nParticles];
//	cudaMalloc((void**)&dev_p, nParticles * sizeof(Particle));
//	CUDA_CHECK_ERROR;
//
//	for (int i = 0; i < nParticles; i++) {
//		cudaMalloc((void**)&(w[i]), 300 * sizeof(double));
//		CUDA_CHECK_ERROR;
//		cudaMemcpy(&(dev_p[i]._w), &(w[i]), sizeof(double *), cudaMemcpyHostToDevice);
//		CUDA_CHECK_ERROR;
//	}
//	double testval = 32.7;
//	cudaMemcpy(w[2] + 2, &testval, sizeof(double), cudaMemcpyHostToDevice);
//	CUDA_CHECK_ERROR;
//	test << <1, 32 >> >(dev_p);
//	cudaDeviceSynchronize();
//	CUDA_CHECK_ERROR;
//	printf("Done!\n");
//}
