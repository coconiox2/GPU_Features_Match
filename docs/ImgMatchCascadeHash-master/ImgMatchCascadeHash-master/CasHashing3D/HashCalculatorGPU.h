#include <stdio.h>
#include <cuda_runtime.h>
#include "Common.h"


void compute_hash_GPU(ImageData **, ImageData *, int , int *, int *);
void compute_hash_GPU_revised(ImageData **, const ImageData *, const int, const cudaPitchedPtr, const int *, const int);
//void call_kernel_hamming(ImageData **device_ptr, ImageData *host_ptr, int img_cnt, float *, float *, float *);
//void compute_hashes_serial(ImageData **device_ptr, ImageData *host_ptr ,int img_cnt, float *, float *, float); 
void compute_hamming_distance_GPU(ImageData *, ImageData *, HashData *, HashData *, int , uint16_t*, uint8_t*);