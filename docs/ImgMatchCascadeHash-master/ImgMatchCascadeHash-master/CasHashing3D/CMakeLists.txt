set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -lpthread")
cmake_minimum_required(VERSION 3.2)
project(CasHashing3D)
find_package(CUDA REQUIRED)
# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -lineinfo -gencode arch=compute_35,code=sm_35;
    )

# Specify the files that need to be compiled and linked.
# This will create an executable named 'CasHashing3D'.
cuda_add_executable(CasHashing3D
	Main.cc
	KeyFileReader.cc	
	HashCalculatorGPU.cu
	TransferData.cc
	HashDataGenerator.cc
	HashDataGeneratorGPU.cu
	BucketBuilder.cc
	ImageMatchPair.cc
	MatchPairGPU.cu
	EuclideanDistanceCalculatorGPU.cu
	BucketBuilderGPU.cu
)

# Activate (and require) C++11 support
set_property(TARGET CasHashing3D PROPERTY CXX_STANDARD 11)
set_property(TARGET CasHashing3D PROPERTY CXX_STANDARD_REQUIRED ON)

# Instruct CMake to set the correct paths in the bsub job script and to copy it
# to the directory that will contain the executable.
configure_file(job.sh.in job.sh @ONLY)
