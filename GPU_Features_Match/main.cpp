#include "third_party/cmdLine/cmdLine.h"
#include "computeMatchesCU.h"


using namespace computeMatches;
int main(int argc, char** argv) {
	CmdLine cmd;

	std::string sSfM_Data_Filename;
	std::string sMatchesDirectory = "";
	std::string sGeometricModel = "f";
	float fDistRatio = 0.8f;
	int iMatchingVideoMode = -1;
	std::string sPredefinedPairList = "";
	std::string sNearestMatchingMethod = "AUTO";
	bool bForce = false;
	bool bGuided_matching = false;
	int imax_iteration = 2048;
	unsigned int ui_max_cache_size = 0;
	
	//required
	cmd.add(make_option('i', sSfM_Data_Filename, "input_file"));
	cmd.add(make_option('o', sMatchesDirectory, "out_dir"));
	// Options
	cmd.add(make_option('r', fDistRatio, "ratio"));
	cmd.add(make_option('g', sGeometricModel, "geometric_model"));
	cmd.add(make_option('v', iMatchingVideoMode, "video_mode_matching"));
	cmd.add(make_option('l', sPredefinedPairList, "pair_list"));
	cmd.add(make_option('n', sNearestMatchingMethod, "nearest_matching_method"));
	cmd.add(make_option('f', bForce, "force"));
	cmd.add(make_option('m', bGuided_matching, "guided_matching"));
	cmd.add(make_option('I', imax_iteration, "max_iteration"));
	cmd.add(make_option('c', ui_max_cache_size, "cache_size"));


	try {
		if (argc == 1) throw std::string("Invalid command line parameter.");
		cmd.process(argc, argv);
	}
	catch (const std::string& s) {
		std::cerr << "Usage: " << argv[0] << '\n'
			<< "[-i|--input_file] a SfM_Data file\n"
			<< "[-o|--out_dir path] output path where computed are stored\n"
			<< "\n[Optional]\n"
			<< "[-f|--force] Force to recompute data]\n"
			<< "[-r|--ratio] Distance ratio to discard non meaningful matches\n"
			<< "   0.8: (default).\n"
			<< "[-g|--geometric_model]\n"
			<< "  (pairwise correspondences filtering thanks to robust model estimation):\n"
			<< "   f: (default) fundamental matrix,\n"
			<< "   e: essential matrix,\n"
			<< "   h: homography matrix.\n"
			<< "   a: essential matrix with an angular parametrization,\n"
			<< "   o: orthographic essential matrix.\n"
			<< "   u: upright essential matrix.\n"
			<< "[-v|--video_mode_matching]\n"
			<< "  (sequence matching with an overlap of X images)\n"
			<< "   X: with match 0 with (1->X), ...]\n"
			<< "   2: will match 0 with (1,2), 1 with (2,3), ...\n"
			<< "   3: will match 0 with (1,2,3), 1 with (2,3,4), ...\n"
			<< "[-l]--pair_list] file\n"
			<< "[-n|--nearest_matching_method]\n"
			<< "  AUTO: auto choice from regions type,\n"
			<< "  For Scalar based regions descriptor:\n"
			<< "    BRUTEFORCEL2: L2 BruteForce matching,\n"
			<< "    HNSWL2: L2 Approximate Matching with Hierarchical Navigable Small World graphs,\n"
			<< "    ANNL2: L2 Approximate Nearest Neighbor matching,\n"
			<< "    CASCADEHASHINGL2: L2 Cascade Hashing matching.\n"
			<< "    FASTCASCADEHASHINGL2: (default)\n"
			<< "      L2 Cascade Hashing with precomputed hashed regions\n"
			<< "     (faster than CASCADEHASHINGL2 but use more memory).\n"
			<< "  For Binary based descriptor:\n"
			<< "    BRUTEFORCEHAMMING: BruteForce Hamming matching.\n"
			<< "[-m|--guided_matching]\n"
			<< "  use the found model to improve the pairwise correspondences.\n"
			<< "[-c|--cache_size]\n"
			<< "  Use a regions cache (only cache_size regions will be stored in memory)\n"
			<< "  If not used, all regions will be load in memory."
			<< std::endl;

		std::cerr << s << std::endl;
		return EXIT_FAILURE;
	}

	std::cout << " You called : " << "\n"
		<< argv[0] << "\n"
		<< "--input_file " << sSfM_Data_Filename << "\n"
		<< "--out_dir " << sMatchesDirectory << "\n"
		<< "Optional parameters:" << "\n"
		<< "--force " << bForce << "\n"
		<< "--ratio " << fDistRatio << "\n"
		<< "--geometric_model " << sGeometricModel << "\n"
		<< "--video_mode_matching " << iMatchingVideoMode << "\n"
		<< "--pair_list " << sPredefinedPairList << "\n"
		<< "--nearest_matching_method " << sNearestMatchingMethod << "\n"
		<< "--guided_matching " << bGuided_matching << "\n"
		<< "--cache_size " << ((ui_max_cache_size == 0) ? "unlimited" : std::to_string(ui_max_cache_size)) << std::endl;

	computeMatches::computeMatchesMVG(sSfM_Data_Filename, sMatchesDirectory, sPredefinedPairList, sGeometricModel);
}
//int sss()
//{
//	//Defines the path and name of the read and output files
//	std::string sInputJpgDir_father;
//	std::string sSfM_Data_FilenameDir_father;
//	std::string sMatchesOutputDir_father;
//
//	CmdLine cmd;
//	cmd.add(make_option('i', sInputJpgDir_father, "input_jpg"));
//	cmd.add(make_option('d', sSfM_Data_FilenameDir_father, "input_sfmData"));
//	cmd.add(make_option('o', sMatchesOutputDir_father, "out_dir"));
//
//
//	try {
//		if (argc == 1) throw std::string("Invalid command line parameter.");
//		cmd.process(argc, argv);
//	}
//	catch (const std::string& s) {
//		std::cerr << "Usage: " << argv[0] << '\n'
//			<< std::endl;
//
//		std::cerr << s << std::endl;
//		return EXIT_FAILURE;
//	}
//
//	//computeMatches::getInputAndOutputDir(sInputJpgDir_father, sSfM_Data_FilenameDir_father, sMatchesOutputDir_father);
//
//
//	openMVG::system::Timer computeHashTimeCost;
//	Eigen::VectorXf _zero_mean_descriptor;
//	computeMatches::computeZeroMeanDescriptors(_zero_mean_descriptor, sInputJpgDir_father, sMatchesOutputDir_father);
//	computeMatches::computeHashes(_zero_mean_descriptor, sInputJpgDir_father, sMatchesOutputDir_father);
//	std::cout << "Tasks (computing hash for all groups) cost " << computeHashTimeCost.elapsed() << "s" << std::endl;
//	openMVG::system::Timer computeMatchesTimeCost;
//	computeMatches::computeMatches(sInputJpgDir_father);
//
//	//computeMatches::test();
//
//
//	std::cout << "group_count: " << group_count << std::endl;
//	std::cout << "block_count_per_group: " << block_count_per_group << std::endl;
//	std::cout << "image_count_per_block: " << image_count_per_block << std::endl;
//	std::cout << "All images count:(group_count*block_count_per_group*image_count_per_block): " << group_count*block_count_per_group*image_count_per_block << std::endl;
//	std::cout << "Tasks (computing matches for all groups) cost: " << computeMatchesTimeCost.elapsed() << "s" << std::endl;
//	//computeMatches::test();
//
//
//	getchar();
//	getchar();
//	getchar();
//	getchar();
//	return 0;
//}