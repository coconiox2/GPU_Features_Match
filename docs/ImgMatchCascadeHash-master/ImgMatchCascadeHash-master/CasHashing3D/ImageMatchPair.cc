#include "ImageMatchPair.h"
#include "EuclideanDistanceCalculatorGPU.h"

//int calculateCandidateSiftPointsCount(ImageData *imgData1, ImageData *imgData2){
//	int totalCandidateSiftPoints = 0;
//	int bucket_id = -1;
//	int countSiftPointsBucket = 0;
//}

void matchImages(ImageData *imageDataList, ImageData **deviceImageDataPointerList, int cntImage, HashData ** hashDataVector, char *outFileName)
{
	
	MatchList matchList;
	FILE* outFile = fopen(outFileName, "w");
	uint16_t* deviceHammingBuckets;
	cudaMalloc(&deviceHammingBuckets, sizeof(uint16_t) * maxCandidatesPerBucket * (maxHammingDistance + 1) * maxPointsPerImage);

	cudaStream_t streams[numberOfStreams];
	
	uint16_t *deviceCandidateList = mallocCandidatePointsArray(maxPointsPerImage);
	uint8_t *deviceCandidateCntList = mallocCandidateCntList(maxPointsPerImage);
			
	int *hostMatchingPoints, *deviceMatchingPoints;
	hostMatchingPoints = (int*)malloc(sizeof(int) * maxPointsPerImage);
	cudaMalloc(&deviceMatchingPoints, sizeof(int) * maxPointsPerImage);
	
			
	for(int i = 0; i < numberOfStreams; i++)
	{
		cudaStreamCreate(&streams[i]);
	}
	
	int pairCount = 0;
	int streamNumber;
	int matchingCount = 0;
	//printf("Matching");
	for (int j = 0; j <= cntImage - 1; j++) {
		//fprintf(stderr, "outer: %d\n", j);
		for (int i = 0; i < j; i++) {
			//fprintf(stderr, "\tinner: %d\n", i);
			matchingCount = 0;
			streamNumber = pairCount % numberOfStreams;
			compute_hamming_distance_GPU(deviceImageDataPointerList[i], deviceImageDataPointerList[j], hashDataVector[i], hashDataVector[j], imageDataList[i].cntPoint, deviceCandidateList, deviceCandidateCntList, deviceHammingBuckets, &streams[streamNumber]);
			compute_euclidean_distance_GPU(deviceImageDataPointerList[i], deviceImageDataPointerList[j], imageDataList[i].cntPoint, deviceCandidateList, deviceCandidateCntList, deviceMatchingPoints, i, &streams[streamNumber]);

			//cudaMemcpyAsync(hostMatchingPoints, deviceMatchingPoints, sizeof(int) * (imageDataList[i].cntPoint), cudaMemcpyDeviceToHost, streams[streamNumber]);
			
			cudaMemcpy(hostMatchingPoints, deviceMatchingPoints, sizeof(int) * (imageDataList[i].cntPoint), cudaMemcpyDeviceToHost);
			
			for (int k = 0; k < imageDataList[i].cntPoint; k++) {
				if (hostMatchingPoints[k] != -1) {
					matchingCount++;
				}				
			}
			
			if (matchingCount > kMinMatchListLen) {
				fprintf(outFile, "%d %d\n%d\n", i, j, matchingCount);
				for (int k = 0; k < imageDataList[i].cntPoint; k++) {

					if (hostMatchingPoints[k] != -1) {
						fprintf(outFile, "%d %d\n", k, hostMatchingPoints[k]);
					}
				}
			}
			pairCount++;
		}
	}
	freeCandidatePointsArray(deviceCandidateList);
	freeCandidateCntList(deviceCandidateCntList);
	cudaFree(deviceMatchingPoints);

	cudaFree(deviceHammingBuckets);
	fclose(outFile);
		
}


int findMinValIndex_host(int data_index, int cntCandidateFound, uint16_t* hostCandidateList, double* candidateDistListTop) {

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
		//printf(" inside if match i:%d, index = %d\n", data_index, minValInd_1);
		return minValInd_1;
	}
	else
		return -1;
}