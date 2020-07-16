#include "Common.h"

void compute_euclidean_distance_GPU(ImageData *, ImageData *, int, uint16_t*, uint8_t*, int*, int, cudaStream_t *);
__device__ int findMinValIndex_device(int, int, uint16_t*, double*);