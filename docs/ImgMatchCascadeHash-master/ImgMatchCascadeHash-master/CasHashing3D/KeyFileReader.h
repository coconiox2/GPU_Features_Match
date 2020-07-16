#pragma once
#include "Common.h"

#include <cstdio>

// this class is used to read SIFT feature data from file
class KeyFileReader
{
public:
    // read SIFT feature data from file; the file path is specified in <imageData>
    static void Read(ImageData& imageData);

private:
    // read SIFT feature data from plain text file
    static void ReadKeys(FILE* fp, ImageData& imageData);
};
