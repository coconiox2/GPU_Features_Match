#pragma once
#include "Common.h"
#include <cuda_runtime.h>
//void copyImageDataToDevice(ImageData* imgArrDev, ImageData* imgArrHost, int size);
void copyImageDataToDevice(ImageData **imgArrDev, const ImageData *imgArrHost, size_t size);
void uploadImageDataToDevice(ImageData *, int);
void projectionMatrixMalloc(int l, int m, int n, int ***mat);
void uploadImagesToDevice(ImageData **d_imgData, ImageData *h_imgData);
void downloadHashData(ImageData *);
HashData* uploadBucket(ImageData *);
uint16_t* mallocCandidatePointsArray(int);
uint8_t* mallocCandidateCntList(int);
void freeSiftData(ImageData *);
void freeCandidatePointsArray(uint16_t* );
void freeCandidateCntList(uint8_t* );