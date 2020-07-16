#include "computeMatchesCU.h"

int main(int argc, char** argv) {
	computeMatches::ComputeMatches sComputeMatches;
	sComputeMatches.computeZeroMeanDescriptors(computeMatches::zero_mean_descriptor);
	sComputeMatches.computeMatches();
	sComputeMatches.test();
	
	
	getchar();
}