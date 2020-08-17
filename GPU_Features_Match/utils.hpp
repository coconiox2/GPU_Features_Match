#pragma once
#ifndef _UTILS_HPP_
#define _UTILS_HPP_
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <fstream>  
#include <iostream>
#include <random>
#include <utility>
#include <map> 
#include <vector>
#include <string>
#include <sstream>  

#include <openMVG/types.hpp>
#include "openMVG/stl/dynamic_bitset.hpp"
#include "computeMatchesCU.h"

using namespace std;
using namespace openMVG;
using namespace computeMatches;

//using BlockType = unsigned char;

namespace hashed_code_file_io {
	
	bool wirte_Hashed_Description
	(
		ofstream &out,
		HashedDescription &sHashedDescription
	)
	{
		
		size_t len_hash_code = sHashedDescription.hash_code.size();
		out.write((char*)&len_hash_code, sizeof(len_hash_code));
		
		for (int i = 0; i < 16; i++) {
			out.write((char*)&sHashedDescription.hash_code.data()[i], 1);
		}
		//out.write((char*)&sHashedDescription.hash_code.data()[0], 16);

		size_t len_bucket_ids = sHashedDescription.bucket_ids.size();
		out.write((char*)&len_bucket_ids, sizeof(len_bucket_ids));
		for (int i = 0; i < len_bucket_ids; i++) {
			out.write((char*)&sHashedDescription.bucket_ids[i], sizeof(uint16_t));
		}
		//out.write((char*)&sHashedDescription.bucket_ids[0], len_bucket_ids * sizeof(uint16_t));

		return true;
	}

	bool read_Hashed_Description
	(
		ifstream &in,
		HashedDescription &sHashedDescription
	)
	{
		

		size_t len_hash_code;
		in.read((char*)&len_hash_code, sizeof(len_hash_code));
		sHashedDescription.hash_code = stl::dynamic_bitset(len_hash_code);

		for (int i = 0; i < 16; i++) {
			in.read((char *)&sHashedDescription.hash_code.data()[i], 1);
		}
		//in.read((char*)&sHashedDescription.hash_code.data()[0], 16);

		size_t len_bucket_ids;
		in.read((char*)&len_bucket_ids, sizeof(len_bucket_ids));
		//len_bucket_ids = 6;
		if (len_bucket_ids != 6) {
			if (in.good())
			{
				cout << "good" << endl;
			}
			if (in.bad())
			{
				cout << "bad" << endl;
			}
			if (in.fail())
			{
				cout << "fail" << endl;
			}
			if (in.eof())
			{
				cout << "eof" << endl;
			}
		}
		sHashedDescription.bucket_ids.resize(len_bucket_ids);
		for (int i = 0; i < len_bucket_ids; i++) {
			in.read((char*)&sHashedDescription.bucket_ids[i], sizeof(uint16_t));
		}
		//in.read((char*)&sHashedDescription.bucket_ids[0], len_bucket_ids * sizeof(uint16_t));
		return true;
	}

	bool write_hashed_descriptions
	(
		ofstream &out,
		HashedDescriptions &sHashedDescriptions
	)
	{
		size_t len_hashed_desc = sHashedDescriptions.hashed_desc.size();
		out.write((char*)&len_hashed_desc, sizeof(len_hashed_desc));

		for (int i = 0; i < sHashedDescriptions.hashed_desc.size(); i++) {
			//std::cout << i << std::endl;
			wirte_Hashed_Description(out, sHashedDescriptions.hashed_desc[i]);
		}

		size_t len_buckets_0 = sHashedDescriptions.buckets.size();
		size_t len_buckets_1 = sHashedDescriptions.buckets[0].size();
		//size_t len_buckets_2 = sHashedDescriptions.buckets[0][0].size();

		out.write((char*)&len_buckets_0, sizeof(len_buckets_0));
		out.write((char*)&len_buckets_1, sizeof(len_buckets_1));
		//out.write((char*)&len_buckets_2, sizeof(len_buckets_2));

		for (int i = 0; i < sHashedDescriptions.buckets.size(); i++) {
			for (int j = 0; j < sHashedDescriptions.buckets[0].size(); j++) {
				size_t len_buckets_2 = sHashedDescriptions.buckets[i][j].size();
				out.write((char*)&len_buckets_2, sizeof(len_buckets_2));
				if (len_buckets_2 != 0) {
					for (int k = 0; k < len_buckets_2; k++) {
						out.write((char*)&sHashedDescriptions.buckets[i][j][k], sizeof(int));
					}
					//out.write((char*)&sHashedDescriptions.buckets[i][j][0], len_buckets_2 * sizeof(int));
				}
			}
		}
		return true;
	}

	bool read_hashed_descriptions
	(
		ifstream &in,
		HashedDescriptions &sHashedDescriptions
	)
	{
		size_t len_hashed_desc;
		in.read((char *)&len_hashed_desc, sizeof(len_hashed_desc));

		sHashedDescriptions.hashed_desc.resize(len_hashed_desc);
		for (int i = 0; i < len_hashed_desc; i++) {
			//std::cout << i << std::endl;
			read_Hashed_Description(in, sHashedDescriptions.hashed_desc[i]);
		}

		size_t len_buckets_0;
		size_t len_buckets_1;
		//size_t len_buckets_2;

		in.read((char*)&len_buckets_0, sizeof(len_buckets_0));
		in.read((char*)&len_buckets_1, sizeof(len_buckets_1));
		//in.read((char*)&len_buckets_2, sizeof(len_buckets_2));

		sHashedDescriptions.buckets.resize(len_buckets_0);
		for (int i = 0; i < len_buckets_0; i++) {
			sHashedDescriptions.buckets[i].resize(len_buckets_1);
			for (int j = 0; j < len_buckets_1; j++) {
				size_t len_buckets_2;
				in.read((char*)&len_buckets_2, sizeof(len_buckets_2));
				sHashedDescriptions.buckets[i][j].resize(len_buckets_2);
				if (len_buckets_2 != 0) {
					for (int k = 0; k < len_buckets_2; k++) {
						in.read((char *)&sHashedDescriptions.buckets[i][j][k], sizeof(int));
					}
					//in.read((char *)&sHashedDescriptions.buckets[i][j][0], len_buckets_2 * sizeof(int));
				}
			}
		}
		return true;
	}

	bool write_hashed_base
	(
		const std::string & sfileNameHash,
		std::map<openMVG::IndexT, HashedDescriptions> &shashed_base_
	)
	{
		ofstream out;
		out.open(sfileNameHash, ios::out|ios::binary);
		if (!out.is_open()) {
			cout << "error when ofstream open the file!" << endl;
			return false;
		}

		size_t len_shashed_base = shashed_base_.size();
		out.write((char*)&len_shashed_base, sizeof(len_shashed_base));

		for (int i = 0; i < shashed_base_.size(); i++) {
			write_hashed_descriptions(out, shashed_base_[i]);
		}
		out.close();
		return true;
	}

	bool read_hashed_base
	(
		const std::string & sfileNameHash,
		std::map<openMVG::IndexT, HashedDescriptions> &shashed_base_
	)
	{
		ifstream in;
		in.open(sfileNameHash, ios::in|ios::binary);
		if (!in.is_open()) {
			cout << "error when ofstream open the file!" << endl;
			return false;
		}

		size_t len_shashed_base;
		in.read((char*)&len_shashed_base, sizeof(len_shashed_base));

		for (int i = 0; i < len_shashed_base; i++) {
			read_hashed_descriptions(in, shashed_base_[i]);
		}
		in.close();
		return true;
	}
}//namespace hashed_code_file_io

#endif // !_UTILS_HPP_


