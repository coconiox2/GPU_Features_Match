#pragma once

#include "Common.h"
#include "BucketBuilderGPU.h"

// this class is used to generate bucket list based on each SIFT point's bucket index
class BucketBuilder
{
public:
	// generate bucket list based on each SIFT point's bucket index
	static void Build(ImageData& imageData,int);
	static void Build_Revised(ImageData **, ImageData&, int, int);
};


