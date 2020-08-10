#include <iostream>
//CUDA V 10.2
#include <math.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

extern "C" int testCUDACPP(int a,int b) {
	int c = a + b;
	return c;
}
