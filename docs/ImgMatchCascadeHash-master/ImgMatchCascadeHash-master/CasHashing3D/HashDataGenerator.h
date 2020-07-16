#pragma once

#include <cstdlib>
#include <cmath>
#include <iostream>
#include "Common.h"

class HashDataGenerator {

public:
	static void generateFirstProjectionMatrix(int *);
	static void generateSecondProjectionMatrix(int *);	
	static void upload3DHashMatrix(cudaPitchedPtr *, int *);
	static void upload2DHashMatrix(int **, const int *, size_t &);
	static float getRandNumber();
};
