#pragma once
#define MAX_CANDIDATE_LIST_SIZE 1500
#define MAX_CANDIDATE_PER_BUCKET
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>

//#define Bucket_SecHash // use secondary hashing function to construct buckets
//#define Bucket_PriHashSel // use selected bits in primary hashing function to construct buckets

#define CUDA_CHECK_ERROR                                                       \
    do {                                                                       \
        const cudaError_t err = cudaGetLastError();                            \
        if (err != cudaSuccess) {                                              \
            const char *const err_str = cudaGetErrorString(err);               \
            std::cerr << "Cuda error in " << __FILE__ << ":" << __LINE__ - 1   \
                      << ": " << err_str << " (" << err << ")" << std::endl;   \
            exit(EXIT_FAILURE);                                                \
        }																	   \
    } while(0)


const int kDimSiftData = 128; // the number of dimensions of SIFT feature
const int kDimHashData = 128; // the number of dimensions of Hash code
const int kBitInCompHash = 64; // the number of Hash code bits to be compressed; in this case, use a <uint64_t> variable to represent 64 bits
const int kDimCompHashData = kDimHashData / kBitInCompHash; // the number of dimensions of CompHash code
const int kMinMatchListLen = 16; // the minimal list length for outputing SIFT matching result between two images
const int maxHammingDistance = 128;
const int kCntBucketBit = 8; // the number of bucket bits
const int kCntBucketGroup = 6; // the number of bucket groups
const int kCntBucketPerGroup = 1 << kCntBucketBit; // the number of buckets in each group

const int kCntCandidateTopMin = 6; // the minimal number of top-ranked candidates
const int kCntCandidateTopMax = 6; // the maximal number of top-ranked candidates

const float matchThreshold = 0.32f; // threshold for selecting sift point in match pair list

typedef int* SiftDataPtr; // SIFT feature is represented with <int> type
typedef uint8_t* HashDataPtr; // Hash code is represented with <uint8_t> type; only the lowest bit is used
typedef uint64_t* CompHashDataPtr; // CompHash code is represented with <uint64_t> type
typedef int* BucketElePtr; // index list of points in a specific bucket

const int numberOfStreams = 3;
const int maxPointsPerImage = 4000;
const int maxCandidatesPerBucket = 150;

typedef struct
{
    int cntPoint; // the number of SIFT points
    char keyFilePath[100]; // the path to SIFT feature file
    SiftDataPtr* siftDataPtrList; // SIFT feature for each SIFT point
	SiftDataPtr deviceSiftDataPtrList; //List of SIFT features on device. Linearized.
    HashDataPtr deviceHashDataPtrList; // Hash code for each SIFT point. Linearized.
	HashDataPtr hashDataPtrList; //Hash code for each SIFT point on host. Linearized.
    CompHashDataPtr compHashDataPtrList; // CompHash code for each SIFT point
    uint16_t* deviceBucketIDSiftPoint; // bucket entries for each SIFT point, linearized
	uint16_t* bucketIDSiftPoint;
	
	
    //uint16_t* device_bucketIDList[kCntBucketGroup];
	int cntSiftPointInBucket[kCntBucketGroup][kCntBucketPerGroup]; // the number of SIFT points in each bucket
	int *deviceCntSiftPointInBucket;
    BucketElePtr bucketList[kCntBucketGroup][kCntBucketPerGroup]; // SIFT point index list for all
	BucketElePtr *deviceBucketList;
}   ImageData; // all information needed for an image to perform CasHash-Matching


typedef struct
{
	int cntPoint; // the number of sift points in image
	int* deviceCntSiftPointInBucket; // the number of SIFT points in bucket
	BucketElePtr *deviceBucketList; 
	
	//HashDataPtr* hashDataPtrList; // Hash code for each SIFT point
	//CompHashDataPtr* compHashDataPtrList; // CompHash code for each SIFT point
	//uint16_t* bucketIDList[kCntBucketGroup]; // bucket entries for each SIFT point
    //int cntEleInBucket[kCntBucketGroup][kCntBucketPerGroup]; // the number of SIFT points in each bucket
	//int *bucket_to_features_map;
}HashData; // the hashes and buckets for each point in the image


typedef std::vector<std::pair<int, int> > MatchList; // SIFT point match list between two images

