	#pragma once
#include <cuda_runtime.h>
#include "Common.h"

#define WARP_SIZE 32

void printFirstProjMatrixGPU(const cudaPitchedPtr);
void printSecondProjMatrixGPU(const int *, const size_t);
