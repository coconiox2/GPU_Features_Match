#include "computeMatchesCU.h"

using namespace computeMatches;
int main(int argc, char** argv) {
	openMVG::system::Timer computeHashTimeCost;

	/*Eigen::VectorXf _zero_mean_descriptor;
	computeMatches::computeZeroMeanDescriptors(_zero_mean_descriptor);
	computeMatches::computeHashes(_zero_mean_descriptor);
	std::cout << "Tasks (computing hash for all groups) cost " << computeHashTimeCost.elapsed() << "s" << std::endl;*/
	openMVG::system::Timer computeMatchesTimeCost;
	computeMatches::computeMatches();
	
	//computeMatches::test();

	
	std::cout << "group_count: " << group_count << std::endl;
	std::cout << "block_count_per_group: " << block_count_per_group << std::endl;
	std::cout << "image_count_per_block: " << image_count_per_block << std::endl;
	std::cout << "All images count:(group_count*block_count_per_group*image_count_per_block): " << group_count*block_count_per_group*image_count_per_block << std::endl;
	std::cout << "Tasks (computing matches for all groups) cost: " << computeMatchesTimeCost.elapsed() << "s" << std::endl;
	//computeMatches::test();
	
	
	getchar();
	getchar();
	getchar();
	getchar();
}