#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<immintrin.h>
#include<stdint.h>
#include<time.h>
#include<emmintrin.h>
#include<math.h>
#include <iostream>
#include <random>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <unordered_map>
#include <xmmintrin.h>  
#include <unordered_set>
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
static std::mt19937_64 rng;  
static bool rng_initialized = false;


#define test_number 50
#define N ((unsigned long)(pow(2, 13.6) + 0.5))   //2^13.6


#define N0 0xFF000000
#define N1 0x00FF0000
#define N2 0x0000FF00
#define N3 0x000000FF





void init_rng() {
	if (!rng_initialized) {
		auto seed = std::chrono::system_clock::now().time_since_epoch().count();
		rng.seed(static_cast<uint64_t>(seed));
		rng_initialized = true;
	}
}


void fill_random(void* ptr, size_t size) {
	init_rng();
	uint8_t* bytes = static_cast<uint8_t*>(ptr);
	for (size_t i = 0; i < size; ) {
		uint64_t rand_val = rng();
		for (int j = 0; j < 8 && i < size; j++, i++) {
			bytes[i] = static_cast<uint8_t>(rand_val >> (8 * j));
		}
	}
}


std::vector<uint16_t> generateUniqueRandomSet() {
	std::vector<uint16_t> result;
	result.reserve(N); 

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<uint16_t> dis(1, 65535);

	std::vector<bool> exists(65536, false); 

	while (result.size() < N) {
		uint16_t num = dis(gen);
		if (!exists[num] && ((num & 0xFF00) != 0) && ((num & 0x00FF) != 0)) {
			exists[num] = true;
			result.push_back(num);
		}
	}

	return result;
}



void encrypt(__m128i plain, const __m128i key[6], __m128i* cipher) {

	__m128i current_state = plain;  
	current_state = _mm_xor_si128(current_state, key[0]);

	for (int k = 1; k <= 4; ++k) {  
		current_state = _mm_aesenc_si128(current_state, key[k]);
	}
	*cipher = _mm_aesenclast_si128(current_state, key[5]);
}


void pp(__m128i* x)
{
	char a[16];
	memcpy(a, x, 16);
	unsigned t;
	for (t = 0; t < 16; t++)
	{
		printf("%02x", a[t] & 0xff);
		if (t == 3 || t == 7 || t == 11)
			printf(", ");
	}
	printf("\n");
}





int main()
{
	uint16_t i, j0, j1;

	time_t starttime, endtime;
	__m128i plain, dif, cipher;
	__m128i plain1, plain2, plain3;
	__m128i invmask0, invmask1, invmask2, invmask3;
	__m128i inv0, inv1, inv2, inv3;
	__m128i key[6];

	uint32_t index0, index1, index2, index3;
	uint8_t con[test_number] = { 0 };
	uint8_t count = 0;


	std::vector<uint32_t> Tab0, Tab1, Tab2, Tab3;
	size_t b, c; 

	invmask0 = _mm_set_epi32(N2, N1, N0, N3);  //nonzero in the 0-th inverse diagonal
	invmask1 = _mm_set_epi32(N1, N0, N3, N2);  //nonzero in the 1-th inverse diagonal
	invmask2 = _mm_set_epi32(N0, N3, N2, N1);  //nonzero in the 2-th inverse diagonal
	invmask3 = _mm_set_epi32(N3, N2, N1, N0);  //nonzero in the 3-th inverse diagonal

	std::unordered_map<uint32_t, std::vector<uint16_t>> Hash0;
	std::unordered_map<uint32_t, std::vector<uint16_t>> Hash1;
	std::unordered_map<uint32_t, std::vector<uint16_t>> Hash2;
	std::unordered_map<uint32_t, std::vector<uint16_t>> Hash3;


	starttime = time(NULL);
	for (i = 0; i < test_number; i++)
	{
		fill_random(&plain, sizeof(plain));

		for (b = 0; b < 6; b++) {
			fill_random(&key[b], sizeof(key[b]));
		}

		std::vector<uint16_t> A0 = generateUniqueRandomSet();
		std::vector<uint16_t> A1 = generateUniqueRandomSet();


		//repeat_pair = 0;
		for (j1 = 0; j1 < N; j1++)    //fix byte five and fifteen
		{

			plain1 = plain;
			//pp(&plain1);
			dif = _mm_set_epi8(((A1[j1] >> 8) & 0xff), 0, 0, 0, 0, 0, 0, 0, 0, 0, (A1[j1] & 0xff), 0, 0, 0, 0, 0);
			plain1 = _mm_and_si128(plain1, _mm_set_epi8(0, -1,  -1,-1,-1,  -1,  -1,  -1,  -1, -1,  0,  -1,  -1,  -1, -1, -1   ));
			plain1 = _mm_xor_si128(plain1, dif);
			//pp(&dif);
			//pp(&plain1);


			for (j0 = 0; j0 < N; j0++)   //byte 0 and 10
			{

				plain2 = plain1;
				dif = _mm_set_epi8(0, 0, 0, 0, 0, ((A0[j0] >> 8) & 0xff), 0, 0, 0, 0, 0, 0, 0, 0, 0, (A0[j0] & 0xff));
				plain2 = _mm_and_si128(plain2, _mm_set_epi8( -1, -1, -1, -1, -1,0, -1, -1, -1, -1,  -1, -1, -1, -1, -1,0));
				plain2 = _mm_xor_si128(plain2, dif);
				plain3 = plain2;
				//pp(&dif);
				//pp(&plain3);


				encrypt(plain3, key, &cipher);
				inv0 = _mm_and_si128(cipher, invmask0);
				inv1 = _mm_and_si128(cipher, invmask1);
				inv2 = _mm_and_si128(cipher, invmask2);
				inv3 = _mm_and_si128(cipher, invmask3);

				index0 = (unsigned)(_mm_extract_epi32(inv0, 0) ^ _mm_extract_epi32(inv0, 1) ^ _mm_extract_epi32(inv0, 2) ^ _mm_extract_epi32(inv0, 3));
				index1 = (unsigned)(_mm_extract_epi32(inv1, 0) ^ _mm_extract_epi32(inv1, 1) ^ _mm_extract_epi32(inv1, 2) ^ _mm_extract_epi32(inv1, 3));
				index2 = (unsigned)(_mm_extract_epi32(inv2, 0) ^ _mm_extract_epi32(inv2, 1) ^ _mm_extract_epi32(inv2, 2) ^ _mm_extract_epi32(inv2, 3));
				index3 = (unsigned)(_mm_extract_epi32(inv3, 0) ^ _mm_extract_epi32(inv3, 1) ^ _mm_extract_epi32(inv3, 2) ^ _mm_extract_epi32(inv3, 3));

				Hash0[index0].push_back(A0[j0]);
				Hash1[index1].push_back(A0[j0]);
				Hash2[index2].push_back(A0[j0]);
				Hash3[index3].push_back(A0[j0]);

			}

			for (const auto& kv : Hash0) {
				const std::vector<uint16_t>& vec0 = kv.second;

				if (vec0.size() >= 2) {
					for (b = 0; b < vec0.size() - 1; b++) {
						for (c = b + 1; c < vec0.size(); c++) {
							{
								Tab0.push_back((vec0[b] << 16) ^ vec0[c]);
								Tab0.push_back((vec0[c] << 16) ^ vec0[b]);
							}
						}
					}
				}
			}


			for (const auto& kv : Hash1) {
				const std::vector<uint16_t>& vec1 = kv.second;

				if (vec1.size() >= 2) {
					for (b = 0; b < vec1.size() - 1; b++) {
						for (c = b + 1; c < vec1.size(); c++) {
							{
								Tab1.push_back((vec1[b] << 16) ^ vec1[c]);
								Tab1.push_back((vec1[c] << 16) ^ vec1[b]);
							}
						}
					}
				}
			}


			for (const auto& kv : Hash2) {
				const std::vector<uint16_t>& vec2 = kv.second;

				if (vec2.size() >= 2) {
					for (b = 0; b < vec2.size() - 1; b++) {
						for (c = b + 1; c < vec2.size(); c++) {
							{
								Tab2.push_back((vec2[b] << 16) ^ vec2[c]);
								Tab2.push_back((vec2[c] << 16) ^ vec2[b]);
							}
						}
					}
				}
			}

			for (const auto& kv : Hash3) {
				const std::vector<uint16_t>& vec3 = kv.second;

				if (vec3.size() >= 2) {
					for (b = 0; b < vec3.size() - 1; b++) {
						for (c = b + 1; c < vec3.size(); c++) {
							{
								Tab3.push_back((vec3[b] << 16) ^ vec3[c]);
								Tab3.push_back((vec3[c] << 16) ^ vec3[b]);
							}
						}
					}
				}
			}

			for (auto& kv : Hash0) kv.second.clear();
			Hash0.clear();
			for (auto& kv : Hash1) kv.second.clear();
			Hash1.clear();
			for (auto& kv : Hash2) kv.second.clear();
			Hash2.clear();
			for (auto& kv : Hash3) kv.second.clear();
			Hash3.clear();


		}
		
		

		for (b = 0; b < Tab0.size() - 1; b++)
		{
			for (c = b + 1; c < Tab0.size(); c++)
			{
				if (Tab0[b] == Tab0[c])
				{
					con[i] = 1;
					break;
				}
			}
			if (con[i] > 0)
				break;
		}

		if (con[i] == 0)
		{
			for (b = 0; b < Tab1.size() - 1; b++)
			{
				for (c = b + 1; c < Tab1.size(); c++)
				{
					if (Tab1[b] == Tab1[c])
					{
						con[i] = 1;
						break;
					}
				}
				if (con[i] > 0)
					break;
			}
		}

		if (con[i] == 0)
		{
			for (b = 0; b < Tab2.size() - 1; b++)
			{
				for (c = b + 1; c < Tab2.size(); c++)
				{
					if (Tab2[b] == Tab2[c])
					{
						con[i] = 1;
						break;
					}
				}
				if (con[i] > 0)
					break;
			}
		}

		if (con[i] == 0)
		{
			for (b = 0; b < Tab3.size() - 1; b++)
			{
				for (c = b + 1; c < Tab3.size(); c++)
				{
					if (Tab3[b] == Tab3[c])
					{
						con[i] = 1;
						break;
					}
				}
				if (con[i] > 0)
					break;
			}
		}


		if (con[i] > 0)
		{
			count++;
		}
		//printf("i=%d,repeat pairs=%d\n", i,repeat_pair);


		Tab0.clear();
		Tab1.clear();
		Tab2.clear();
		Tab3.clear();
	}
	printf("there are %d experiments in total, where %d results is 5-round AES\n\n", test_number, count);
	endtime = time(NULL);
	printf("time=%f\n", difftime(endtime, starttime));
	return count;

}





