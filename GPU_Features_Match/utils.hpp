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

#include "openMVG/stl/dynamic_bitset.hpp"
#include "computeMatchesCU.h"

using namespace std;
using namespace computeMatches;



namespace hashed_code_file_io {
	
	bool wirte_Hashed_Description
	(
		ofstream &out,
		HashedDescription &sHashedDescription
	)
	{
		size_t len_hash_code = sHashedDescription.hash_code.size();
		out.write((char*)&len_hash_code, sizeof(len_hash_code));
		out.write((char*)&sHashedDescription.hash_code[0], len_hash_code * sizeof(sHashedDescription.hash_code[0]));

		size_t len_bucket_ids = sHashedDescription.bucket_ids.size();
		out.write((char*)&len_bucket_ids, sizeof(len_bucket_ids));
		out.write((char*)&sHashedDescription.bucket_ids[0], len_bucket_ids * sizeof(uint16_t));

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
		sHashedDescription.hash_code = stl::dynamic_bitset(descriptionDimension);
		in.read((char*)&sHashedDescription.hash_code[0], descriptionDimension * sizeof(sHashedDescription.hash_code[0]));

		size_t len_bucket_ids;
		in.read((char*)&len_bucket_ids, sizeof(len_bucket_ids));
		sHashedDescription.bucket_ids.resize(len_bucket_ids);
		in.read((char*)&sHashedDescription.bucket_ids[0], len_bucket_ids * sizeof(uint16_t));
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
			wirte_Hashed_Description(out, sHashedDescriptions.hashed_desc[i]);
		}

		size_t len_buckets_0 = sHashedDescriptions.buckets.size();
		size_t len_buckets_1 = sHashedDescriptions.buckets[0].size();
		size_t len_buckets_2 = sHashedDescriptions.buckets[0][0].size();

		out.write((char*)&len_buckets_0, sizeof(len_buckets_0));
		out.write((char*)&len_buckets_1, sizeof(len_buckets_1));
		out.write((char*)&len_buckets_2, sizeof(len_buckets_2));

		for (int i = 0; i < sHashedDescriptions.buckets.size(); i++) {
			for (int j = 0; j < sHashedDescriptions.buckets[0].size(); j++) {
				out.write((char*)&sHashedDescriptions.buckets[i][j][0], len_buckets_2*sizeof(int));
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
			read_Hashed_Description(in, sHashedDescriptions.hashed_desc[i]);
		}

		size_t len_buckets_0;
		size_t len_buckets_1;
		size_t len_buckets_2;

		in.read((char*)&len_buckets_0, sizeof(len_buckets_0));
		in.read((char*)&len_buckets_1, sizeof(len_buckets_1));
		in.read((char*)&len_buckets_2, sizeof(len_buckets_2));

		sHashedDescriptions.buckets.resize(len_buckets_0);
		for (int i = 0; i < len_buckets_0; i++) {
			sHashedDescriptions.buckets[i].resize(len_buckets_1);
			for (int j = 0; j < len_buckets_1; j++) {
				sHashedDescriptions.buckets[i][j].resize(len_buckets_2);
				in.read((char *)&sHashedDescriptions.buckets[i][j][0], len_buckets_2 * sizeof(int));
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
		ofstream out(sfileNameHash);
		if (!out.is_open()) {
			cout << "error when ofstream open the file!" << endl;
			return false;
		}

		size_t len_shashed_base = shashed_base_.size();
		out.write((char*)&len_shashed_base, sizeof(len_shashed_base));

		for (int i = 0; i < shashed_base_.size(); i++) {
			write_hashed_descriptions(out, shashed_base_[i]);
		}

		return true;
	}

	bool read_hashed_base
	(
		const std::string & sfileNameHash,
		std::map<openMVG::IndexT, HashedDescriptions> &shashed_base_
	)
	{
		ifstream in(sfileNameHash);
		if (!in.is_open()) {
			cout << "error when ofstream open the file!" << endl;
			return false;
		}

		size_t len_shashed_base;
		in.read((char*)&len_shashed_base, sizeof(len_shashed_base));

		for (int i = 0; i < len_shashed_base; i++) {
			read_hashed_descriptions(in, shashed_base_[i]);
		}
		return true;
	}
}//namespace hashed_code_file_io

#endif // !_UTILS_HPP_


