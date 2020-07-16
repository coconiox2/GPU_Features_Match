#include "HashDataGenerator.h"

void HashDataGenerator::generateFirstProjectionMatrix(int *proMat)
{
	//printf("Serial print\n");
	for (int i = 0; i < kCntBucketGroup; i++) {
		for (int j = 0; j < kCntBucketBit; j++) {
			for (int k = 0; k < kDimSiftData; k++) {
				*(proMat + i * kDimSiftData * kCntBucketBit + j * kDimSiftData + k) = static_cast<int>(getRandNumber() * 1000);
				//std::cout << *(proMat + i * kDimSiftData * kCntBucketBit + j * kDimSiftData + k) << std::endl;
				//std::cout << "index[" << i * kDimSiftData * kCntBucketBit + j * kDimSiftData + k << "]" << proMat[i * kDimSiftData * kCntBucketBit + j * kDimSiftData + k] << std::endl;
			}
		}
	}	
}

void HashDataGenerator::generateSecondProjectionMatrix(int *proMat)
{
	//printf("printing from serial cpu\n");
	for (int i = 0; i < kDimSiftData; i++) {
		for (int j = 0; j < kDimSiftData; j++) {
			*(proMat + (i * kDimSiftData + j)) = static_cast<int>(getRandNumber() * 1000);
			//std::cout << *(proMat + (i * kDimSiftData + j)) << std::endl;
		}
	}
	
}

void HashDataGenerator::upload3DHashMatrix(cudaPitchedPtr *devPitchedPtr, int *firstProjMat)
{
	cudaExtent extent = make_cudaExtent(kDimSiftData * sizeof(int), kCntBucketBit, kCntBucketGroup);

	cudaMalloc3D(devPitchedPtr, extent);
	cudaMemcpy3DParms myParms = { 0 };
	myParms.srcPtr.ptr = firstProjMat;
	myParms.srcPtr.pitch = kDimSiftData * sizeof(int);
	myParms.srcPtr.xsize = kDimSiftData;
	myParms.srcPtr.ysize = kCntBucketBit;
	myParms.dstPtr.ptr = (*devPitchedPtr).ptr;
	myParms.dstPtr.pitch =(*devPitchedPtr).pitch;
	myParms.dstPtr.xsize = kDimSiftData;
	myParms.dstPtr.ysize = kCntBucketBit;
	myParms.extent.width = kDimSiftData * sizeof(int);
	myParms.extent.height = kCntBucketBit;
	myParms.extent.depth = kCntBucketGroup;
	myParms.kind = cudaMemcpyHostToDevice;

	cudaMemcpy3D(&myParms);

}

void HashDataGenerator::upload2DHashMatrix(int **d_secondProjMat, const int *secondProjMat, size_t &pitch)
{
	cudaMallocPitch(d_secondProjMat, &pitch, kDimSiftData * sizeof(int), kDimSiftData);
	cudaMemcpy2D(*d_secondProjMat, pitch, secondProjMat, kDimSiftData * sizeof(int), kDimSiftData * sizeof(int), kDimSiftData, cudaMemcpyHostToDevice);
}


float HashDataGenerator::getRandNumber()
{
	// based on Box-Muller transform; for more details, please refer to the following WIKIPEDIA website:
	// http://en.wikipedia.org/wiki/Box_Muller_transform
	double u1 = (rand() % 1000 + 1) / 1000.0;
	double u2 = (rand() % 1000 + 1) / 1000.0;

	double randVal = sqrt(-2 * log(u1)) * cos(2 * acos(-1.0) * u2);

	return (float)randVal;
	
}




