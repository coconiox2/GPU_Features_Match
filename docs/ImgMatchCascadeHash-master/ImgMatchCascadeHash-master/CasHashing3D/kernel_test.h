#include <stdio.h>
#include <cuda_runtime.h>
#include "Common.h"

//__global__ void hello(ImageData *ptr);
void compute_hash_GPU(ImageData **device_ptr, ImageData *host_ptr ,int img_cnt, float *, float *, float *);
//void call_kernel_hamming(ImageData **device_ptr, ImageData *host_ptr, int img_cnt, float *, float *, float *);
//void call_kernel_test2();
//void call_kernel_test3();
//void compute_hashes_serial(ImageData **device_ptr, ImageData *host_ptr ,int img_cnt, float *, float *, float); 
void compute_hamming_distance_GPU(ImageData *, ImageData *, HashData *, HashData *, int , uint16_t*, uint8_t*);