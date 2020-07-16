//#include <pthread.h>
#include <iostream>
#include <cstring>
//#include <sched.h>
#include <stdlib.h>
//#include <unistd.h>
//#include <curand.h>
//#include <curand_kernel.h>
#include <cuda_profiler_api.h>
#include "Common.h"
#include "KeyFileReader.h"
#include "HashDataGenerator.h"
#include "HashCalculatorGPU.h"
#include "TransferData.h"
#include "Timer.h"
#include "HashDataGeneratorGPU.h"
#include "BucketBuilder.h"
#include "BucketBuilderGPU.h"
#include "ImageMatchPair.h"
using namespace std;

vector<HashData*> hashDataVector;
vector<ImageData> stImageDataList;
ImageData *h_imgData;
ImageData **d_imgDataArray;

double getUnixTime(void)
{
	struct timespec tv;

	if (clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;

	return (tv.tv_sec + (tv.tv_nsec / 1000000000.0));
}

int main(int argc, char* argv[])
{
	// required input parameters list:
	// 1. file path to read all SIFT feature file pathes; one image, one file
	// 2. file path to write SIFT points match list
	// 3. file path to read all image pairs to be matched (optional)
	if (argc != 3) // validate the input parameters
	{
		cout << "Usage: " << argv[0] << "<list.txt> <outfile>" << endl;
		return -1;
	}

	cudaFree(0);
	
	ImageData imageDataNew; // temporal variable to read image data from file
	FILE* inFile = fopen(argv[1], "r");
	
	while (fgets(imageDataNew.keyFilePath, 100, inFile) != NULL) // read the file path which stores SIFT feature data
	{
		imageDataNew.keyFilePath[strlen(imageDataNew.keyFilePath) - 1] = '\0'; // replace the last character in the file path
		stImageDataList.push_back(imageDataNew); // add new element to <stImageDataList>
	}

	fclose(inFile);
	
	int cntImage = static_cast<int>(stImageDataList.size()); // the total number of images
	for (int imageIndex = 0; imageIndex < cntImage; imageIndex++){
		KeyFileReader::Read(stImageDataList[imageIndex]); // read SIFT feature data from file for each image
	}

	double start_time = getUnixTime();

	//Copy the hashes struct to device memory
		
	int *firstProjMat = new int[kCntBucketGroup * kCntBucketBit * kDimSiftData];
	int *secondProjMat = new int[kDimSiftData * kDimSiftData];
	int *d_firstProjMat, *d_secondProjMat1, *d_secondProjMat2;

	double start_time_genprojmatrix = getUnixTime();

	HashDataGenerator::generateSecondProjectionMatrix(secondProjMat);
	HashDataGenerator::generateFirstProjectionMatrix(firstProjMat);
	
	cudaPitchedPtr devPitchedPtr;
	HashDataGenerator::upload3DHashMatrix(&devPitchedPtr, firstProjMat);
	
	size_t pitch;
	HashDataGenerator::upload2DHashMatrix(&d_secondProjMat2, secondProjMat, pitch);
	double stop_time_genprojmatrix = getUnixTime();
	std::cout << "Total Time to gen proj matrix & upload to GPU: " << stop_time_genprojmatrix - start_time_genprojmatrix << std::endl;
	
	double start_time_afterSiftPointUpload = getUnixTime();
	ImageData **device_imgDataPtrList = (ImageData**)malloc(sizeof(ImageData*) * cntImage);
	for (int i = 0; i < cntImage; i++){
		uploadImageDataToDevice(&(stImageDataList[i]), i);
		uploadImagesToDevice(&device_imgDataPtrList[i], &stImageDataList[i]);
	}	
	
	double end_time_afterSiftPointUpload = getUnixTime();
	std::cout << "Total Time to transfer sift data to GPU: " << end_time_afterSiftPointUpload - start_time_afterSiftPointUpload << std::endl;
	
	double start_time_hashing = getUnixTime();

	compute_hash_GPU_revised(device_imgDataPtrList, &stImageDataList[0], cntImage, devPitchedPtr, d_secondProjMat2, pitch);
	
	double stop_time_hashing = getUnixTime();
	std::cout << "Total Time calculate hashes GPU: " << stop_time_hashing - start_time_hashing << std::endl;
	
	for (int image_index = 0; image_index < cntImage; image_index++) {
		downloadHashData(&(stImageDataList[image_index]));
	}

	double start_time_bucket_building = getUnixTime();
	
	for (int imageIndex = 0; imageIndex < cntImage; imageIndex++) {
		BucketBuilder::Build(stImageDataList[imageIndex], imageIndex); // construct multiple groups of buckets for each image
		HashData *deviceHashData = uploadBucket(&(stImageDataList[imageIndex]));
		hashDataVector.push_back(deviceHashData);
	}

	//for (int imageIndex = 0; imageIndex < cntImage; imageIndex++) {
	//	BucketBuilder::Build_Revised(&device_imgDataPtrList[imageIndex], stImageDataList[imageIndex], stImageDataList[imageIndex].cntPoint, imageIndex); // construct multiple groups of buckets for each image
	//	HashData *deviceHashData = uploadBucket(&(stImageDataList[imageIndex]));
	//	hashDataVector.push_back(deviceHashData);
	//}
	

	double end_time_bucket_building = getUnixTime();
	std::cout << "Total Time build buckets GPU: " << end_time_bucket_building - start_time_bucket_building << std::endl;
	
	double start_time_matchingtotal = getUnixTime();
	matchImages(&stImageDataList[0], &device_imgDataPtrList[0], cntImage, &(hashDataVector[0]), argv[2]);
	
	double end_time_bucket_matchingtotal = getUnixTime();
	std::cout << "Total Time to run total image matching on GPU: " << end_time_bucket_matchingtotal - start_time_matchingtotal << std::endl;

	double end_time = getUnixTime();
	std::cout << "Total Time to run on GPU: " << end_time - start_time << std::endl;

	delete[] firstProjMat;
	delete[] secondProjMat;
	cudaFree(d_secondProjMat1);
	
	return 0;
}
