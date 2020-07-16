#include <iostream>
#include <thrust/sort.h>
//#include <inttypes.h>
#include "HashCalculatorGPU.h"
#include "Common.h"

#define BLOCK_SIZE 32
#define TOP_PER_THREAD_HAMMING_LIST_SIZE 32 * 10

__global__ void compute_hash_kernel(ImageData *ptr, int *d_firstProjMat, int *d_secondProjMat, int imageIndex)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < ptr->cntPoint) {
		float sumFirstHash;			
		for (int m = 0; m < kCntBucketGroup; m++){
			int bucketid = 0;
			for (int j = 0; j < kCntBucketBit; j++) {
				sumFirstHash = 0.0f;
				for (int k = 0; k < kDimSiftData; k++) {
					sumFirstHash += ptr->deviceSiftDataPtrList[index * kDimSiftData + k] * d_firstProjMat[m * kDimSiftData * kCntBucketBit + j * kDimSiftData + k];
										
				}
				bucketid = (bucketid << 1) + (sumFirstHash > 0 ? 1 : 0);
				
			}
			
			ptr->deviceBucketIDSiftPoint[ptr->cntPoint * m + index] = bucketid;
			
		}

		float sumSecondHash;
		for (int m = 0; m < kDimSiftData; m++) {
			sumSecondHash = 0.0f;
			for (int j = 0; j < kDimSiftData; j++) {
				sumSecondHash += ptr->deviceSiftDataPtrList[index * kDimSiftData + j] * d_secondProjMat[m * kDimSiftData + j];
			}
			ptr->deviceHashDataPtrList[index * kDimSiftData + m] = (sumSecondHash > 0 ? 1 : 0);

			
		}
			for (int dimCompHashIndex = 0; dimCompHashIndex < kDimCompHashData; dimCompHashIndex++)
			{
				uint64_t compHashBitVal = 0;
				int dimHashIndexLBound = dimCompHashIndex * kBitInCompHash;
				int dimHashIndexUBound = (dimCompHashIndex + 1) * kBitInCompHash;
				for (int dimHashIndex = dimHashIndexLBound; dimHashIndex < dimHashIndexUBound; dimHashIndex++)
				{
					compHashBitVal = (compHashBitVal << 1) + ptr->deviceHashDataPtrList[index * kDimSiftData + dimHashIndex]; // set the corresponding bit to 1/0

				}
				ptr->compHashDataPtrList[index * kDimCompHashData + dimCompHashIndex] = compHashBitVal;
			}
					
	}
}
 
void compute_hash_GPU(ImageData **device_ptr, ImageData *host_ptr ,int img_cnt, int *d_firstProjMat, int *d_secondProjMat)
{
    
    
	for(int i = 0; i < img_cnt; i++){
		
		
		dim3 block(1024);
		dim3 grid((host_ptr->cntPoint + block.x - 1) / block.x);

		compute_hash_kernel<<<grid, block>>>(device_ptr[i], d_firstProjMat, d_secondProjMat, i);
		host_ptr++;
	}
	cudaDeviceSynchronize();
}

__global__ void compute_hash_kernel_revised(ImageData *ptr, const cudaPitchedPtr devPitchedPtr, const int *d_secondProjMat, const int pitch, int imageIndex)
{
	int index = blockIdx.x;
	int tid = threadIdx.x;

	__shared__ float sumFirstHash[kDimSiftData];
	__shared__ float sumSecondHash[kDimSiftData];

	if (index < ptr->cntPoint) {

		//Calculate First Hash Values
		char* devptr = (char*)devPitchedPtr.ptr;
		size_t pitch = devPitchedPtr.pitch;
		size_t slicepitch = pitch * kCntBucketBit;

		for (int d = 0; d < kCntBucketGroup; d++) {
			uint16_t bucketid = 0;
			char *slice = devptr + d * slicepitch;

			for (int y = 0; y < kCntBucketBit; y++) {
				int *row = (int*)(slice + y * pitch);
				sumFirstHash[tid] = ptr->deviceSiftDataPtrList[index * kDimSiftData + tid] * (*(row + tid));
				__syncthreads();

				for (int stride = kDimSiftData / 2; stride > 0; stride >>= 1) {
					if (tid < stride) {
						sumFirstHash[tid] += sumFirstHash[tid + stride];
					}
					__syncthreads();
				}
				if (tid == 0) {
					bucketid = (bucketid << 1) + (sumFirstHash[0] > 0 ? 1 : 0);
					ptr->deviceBucketIDSiftPoint[ptr->cntPoint * d + index] = bucketid;
				}
			}
		}
		
		//Calculate Second Hash Values
		for (int m = 0; m < kDimSiftData; m++) {
			int *matElement = (int*)((char*)d_secondProjMat + m * pitch);
				
			sumSecondHash[tid] = ptr->deviceSiftDataPtrList[index * kDimSiftData + tid] * (*(matElement + tid));
			__syncthreads();

			for (int stride = kDimSiftData / 2; stride > 0; stride >>= 1) {
				if (tid < stride) {
					sumSecondHash[tid] += sumSecondHash[tid + stride];
				}
				__syncthreads();
			}
			if (tid == 0) {
				ptr->deviceHashDataPtrList[index * kDimSiftData + m] = (sumSecondHash[0] > 0 ? 1 : 0);
			}			
		}
		
		// calculate the CompHash code
		// compress <kBitInCompHash> Hash code bits within a single <uint64_t> variable

		if (tid == 0) {
			for (int dimCompHashIndex = 0; dimCompHashIndex < kDimCompHashData; dimCompHashIndex++) {
				uint64_t compHashBitVal = 0;
				int dimHashIndexLBound = dimCompHashIndex * kBitInCompHash;
				int dimHashIndexUBound = (dimCompHashIndex + 1) * kBitInCompHash;

				for (int dimHashIndex = dimHashIndexLBound; dimHashIndex < dimHashIndexUBound; dimHashIndex++) {
					compHashBitVal = (compHashBitVal << 1) + ptr->deviceHashDataPtrList[index * kDimHashData + dimHashIndex]; // set the corresponding bit to 1/0
				}
				ptr->compHashDataPtrList[index * kDimCompHashData + dimCompHashIndex] = compHashBitVal;
			}
		}
	}
}

void compute_hash_GPU_revised(ImageData **device_ptr, const ImageData *host_ptr, const int img_cnt, const cudaPitchedPtr devPitchedPtr, const int *d_secondProjMat, const int pitch)
{

	for (int i = 0; i < img_cnt; i++) {
		dim3 block(128);
		dim3 grid((host_ptr->cntPoint * block.x + block.x - 1) / block.x);

		compute_hash_kernel_revised <<<grid, block>>> (device_ptr[i], devPitchedPtr, d_secondProjMat, pitch, i);
		host_ptr++;
	}
	cudaDeviceSynchronize();
}

