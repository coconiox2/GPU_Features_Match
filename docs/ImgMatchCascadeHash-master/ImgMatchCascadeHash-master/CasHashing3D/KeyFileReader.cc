#include "KeyFileReader.h"
#include <stdio.h>
#include <errno.h>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstdlib>

using namespace std;
void KeyFileReader::Read(ImageData& imageData)
{
    FILE *fpPlain = fopen(imageData.keyFilePath, "r"); // try to open the plain text file

	int errnum;
    if (fpPlain != NULL)
    {
        // if the plain text file has been opened successfully
        ReadKeys(fpPlain, imageData); // read SIFT feature data from the plain text file
        fclose(fpPlain);		
    }
	else
	{
		cout << "Cannot Open File: '" << imageData.keyFilePath << "'" <<endl;
	}
}

void KeyFileReader::ReadKeys(FILE* fp, ImageData& imageData)
{
	//printf("Reading keys\n");
    int cntSiftData; // number of SIFT feature vectors
    int dimSiftData; // dimension of each SIFT feature vector

    // validate the file header
    if (fscanf(fp, "%d%d", &cntSiftData, &dimSiftData) != 2)
    {
        std::cout << "Invalid keypoint file" << std::endl;
        return;
    }
    if (dimSiftData != kDimSiftData) {
        std::cout << "Keypoint descriptor length invalid (should be " << kDimSiftData << ")." << std::endl;
        return;
    }

    // allocate space for SIFT feature vector pointers
    imageData.cntPoint = cntSiftData;
	//printf("printing count %d\n");
    //imageData.siftDataPtrList = (
	//cudaMallocHost(&imageData.siftDataPtrList, sizeof(SiftDataPtr*) * cntSiftData);
	imageData.siftDataPtrList = (SiftDataPtr*)malloc(sizeof(SiftDataPtr*) * cntSiftData);

    // read SIFT feature vectors
    for (int dataIndex = 0; dataIndex < imageData.cntPoint; dataIndex++)
    {
        if (fscanf(fp, "%*f%*f%*f%*f") != 0) // abandon header of each SIFT data
        {
            exit(-1);
        }

        // allocate space for each SIFT feature vector
        imageData.siftDataPtrList[dataIndex] = (int*)malloc(sizeof(int) * kDimSiftData);
        for (int dimIndex = 0; dimIndex < kDimSiftData; dimIndex++)
        {
            if (fscanf(fp, "%d", imageData.siftDataPtrList[dataIndex] + dimIndex) != 1)
            {
                exit(-1);
            }
        }
    }
}
