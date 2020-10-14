#include "third_party/cmdLine/cmdLine.h"
#include "computeMatchesCU.h"


using namespace computeMatches;
int main(int argc, char** argv) {
	//Defines the path and name of the read and output files
	std::string sInputJpgDir_father;
	std::string sSfM_Data_FilenameDir_father;
	std::string sMatchesOutputDir_father;

	CmdLine cmd;
	cmd.add(make_option('i', sInputJpgDir_father, "input_jpg"));
	cmd.add(make_option('d', sSfM_Data_FilenameDir_father, "input_sfmData"));
	cmd.add(make_option('o', sMatchesOutputDir_father, "out_dir"));


	try {
		if (argc == 1) throw std::string("Invalid command line parameter.");
		cmd.process(argc, argv);
	}
	catch (const std::string& s) {
		std::cerr << "Usage: " << argv[0] << '\n'
			<< std::endl;

		std::cerr << s << std::endl;
		return EXIT_FAILURE;
	}
	
	//computeMatches::getInputAndOutputDir(sInputJpgDir_father, sSfM_Data_FilenameDir_father, sMatchesOutputDir_father);
	
	
	openMVG::system::Timer computeHashTimeCost;
	Eigen::VectorXf _zero_mean_descriptor;
	computeMatches::computeZeroMeanDescriptors(_zero_mean_descriptor, sInputJpgDir_father, sMatchesOutputDir_father);
	computeMatches::computeHashes(_zero_mean_descriptor, sInputJpgDir_father, sMatchesOutputDir_father);
	std::cout << "Tasks (computing hash for all groups) cost " << computeHashTimeCost.elapsed() << "s" << std::endl;
	openMVG::system::Timer computeMatchesTimeCost;
	computeMatches::computeMatches(sInputJpgDir_father);
	
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