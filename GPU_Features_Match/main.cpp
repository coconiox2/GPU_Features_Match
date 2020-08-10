#include "computeMatchesCU.h"

int main(int argc, char** argv) {
	
	Eigen::VectorXf _zero_mean_descriptor;
	computeMatches::computeZeroMeanDescriptors(_zero_mean_descriptor);
	computeMatches::computeHashes(_zero_mean_descriptor);
	computeMatches::computeMatches();
	computeMatches::test();
	
	
	getchar();
}