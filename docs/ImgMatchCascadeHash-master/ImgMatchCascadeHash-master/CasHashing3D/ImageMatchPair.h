#include <stdlib.h>
#include <stdio.h>
#include "Common.h"
#include "TransferData.h"
#include "MatchPairGPU.h"

void matchImages(ImageData *, ImageData **, int, HashData **, char *);
void matchImagePair(ImageData *, ImageData *);
int findMinValIndex_host(int, int , uint16_t* , double* );

