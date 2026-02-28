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

#define test_number 200
#define N 65536  //2^16


#define N0 0xFF000000
#define N1 0x00FF0000
#define N2 0x0000FF00
#define N3 0x000000FF


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

bool is_equal(__m128i a, __m128i b) {
	__m128i neq = _mm_xor_si128(a, b);       
	return _mm_test_all_zeros(neq, neq);    
}

unsigned judge(__m128i x, __m128i y, uint8_t i)   //judge whether x+y is zero in the i-th inverse diagonal 
{
	__m128i invmask0 = _mm_set_epi32(N2, N1, N0, N3);  //nonzero in the 0-th inverse diagonal
	__m128i invmask1 = _mm_set_epi32(N1, N0, N3, N2);  //nonzero in the 1-th inverse diagonal
	__m128i invmask2 = _mm_set_epi32(N0, N3, N2, N1);  //nonzero in the 2-th inverse diagonal
	__m128i invmask3 = _mm_set_epi32(N3, N2, N1, N0);  //nonzero in the 3-th inverse diagonal
	__m128i temp = _mm_xor_si128(x, y);
	__m128i inv;
	if (i == 0)
	{
		inv = _mm_and_si128(temp, invmask0);
		return _mm_test_all_zeros(inv, inv);
	}
	else if (i == 1)
	{
		inv = _mm_and_si128(temp, invmask1);
		return _mm_test_all_zeros(inv, inv);
	}
	else if (i == 2)
	{
		inv = _mm_and_si128(temp, invmask2);
		return _mm_test_all_zeros(inv, inv);
	}
	else if (i == 3)
	{
		inv = _mm_and_si128(temp, invmask3);
		return _mm_test_all_zeros(inv, inv);
	}
}


void encrypt(__m128i plain, __m128i key[6], __m128i* cipher)   //5-round encryption
{
	unsigned k;
	__m128i plain0;
	memcpy(&plain0, &plain, sizeof(__m128i));
	plain0 = _mm_xor_si128(plain0, key[0]);
	for (k = 1; k < 5; k++)
	{
		*cipher = _mm_aesenc_si128(plain0, key[k]);
		memcpy(&plain0, cipher, sizeof(__m128i));
	}
	*cipher = _mm_aesenclast_si128(plain0, key[5]);

}


void printseq(unsigned long long a[4])
{
	int i;
	for (i = 0; i < 4; i++)
	{
		printf("%llx,", a[i]);
	}
	printf("\n");
}

void cyc_shift(unsigned long long *a, int r)   
{
	unsigned long long temp1, temp2;
	temp1 = a[0] << r;
	temp2 = a[1] << r;
	//printf("%16x\n", temp1);
	//printf("%16x\n", temp2);
	//printf("%16x\n", a[0] >> (64 - r));
	//printf("%16x\n", a[1] >> (64 - r));
	temp1 = temp1 ^ (a[1] >> (64 - r));
	temp2 = temp2 ^ (a[0] >> (64 - r));
	a[0] = temp1;
	a[1] = temp2;
}

void shift(unsigned long long* a, int r)  
{
	unsigned long long temp1, temp2;
	temp1 = a[0] << r;
	temp2 = a[1] << r;
	temp1 = temp1 ^ (a[1] >> (64 - r));
	a[0] = temp1;
	a[1] = temp2;
}

void m_seq(unsigned long long* a)   //generate m-sequence
{
	unsigned long long temp1 = a[2], temp2 = a[3];
	int r1 = 5, r2 = 17;
	cyc_shift(a, r1);
	shift(&a[2], r2);
	a[2] = a[0] ^ temp1 ^ a[2];
	a[3] = a[1] ^ temp2 ^ a[3];
	a[0] = temp1;
	a[1] = temp2;
}






int main()
{
	unsigned i, j0, j1;
	unsigned long long w;
	time_t starttime, endtime;
	__m128i plain, dif, cipher;
	__m128i plain1, plain2, plain3, plain4, cipher3,cipher4;
	__m128i invmask0, invmask1, invmask2, invmask3;
	__m128i inv0, inv1, inv2, inv3;
	__m128i key[6];
	unsigned long long* a;
	//printf("%d,\n", sizeof(unsigned long long));
	a = (unsigned long long*)malloc(sizeof(unsigned long long) * 4);
	uint32_t index0, index1, index2, index3;
	char con[test_number] = { 0 };
	uint8_t count = 0;


	std::unordered_map<uint32_t, std::vector<__m128i>> Hash0;
	std::unordered_map<uint32_t, std::vector<__m128i>> Hash1;
	std::unordered_map<uint32_t, std::vector<__m128i>> Hash2;
	std::unordered_map<uint32_t, std::vector<__m128i>> Hash3;	


	__m128i* Tab1, * Tab2, * Tab3, * Tab0;
	Tab0 = (__m128i*)malloc(sizeof(__m128i) * 32768); //2^10
	Tab1 = (__m128i*)malloc(sizeof(__m128i) * 32768); //2^10
	Tab2 = (__m128i*)malloc(sizeof(__m128i) * 32768); //2^10
	Tab3 = (__m128i*)malloc(sizeof(__m128i) * 32768); //2^10
	uint32_t m0,  m1,  m2,  m3;
	uint64_t b, c;

	invmask0 = _mm_set_epi32(N2, N1, N0, N3);   //nonzero in the 0-th inverse diagonal
	invmask1 = _mm_set_epi32(N1, N0, N3, N2); //nonzero in the 1-th inverse diagonal
	invmask2 = _mm_set_epi32(N0, N3, N2, N1); //nonzero in the 2-th inverse diagonal
	invmask3 = _mm_set_epi32(N3, N2, N1, N0); //nonzero in the 3-th inverse diagonal


	starttime = time(NULL);
	for (i = 0; i < test_number; i++)
	{
		for (w = 0; w < 4; w++)
		{
			_rdrand64_step(a + w);
		}

		for (w = 0; w < 6; w++)   
		{
			m_seq(a);
			memcpy(&key[w], &a[2], sizeof(__m128i));
		}



		for (j0 = 0; j0 < 32; j0++)  //2^5 plaintext structures
		{
			m_seq(a);
			memcpy(&plain, &a[2], sizeof(__m128i));
			m0 = 0;
			m1 = 0;
			m2 = 0;
			m3 = 0;


			for (j1 = 0; j1 < N; j1++)  //2^16
			{
				dif = _mm_set_epi16(0, 0, (j1 >> 8) & 0x00ff, 0, 0, 0, 0, j1 & 0x00ff);
				plain1 = _mm_xor_si128(plain, dif);
				memcpy(&plain2, &plain1, sizeof(__m128i));

				encrypt(plain2, key, &cipher);

				//pp(&cipher);
				inv0 = _mm_and_si128(cipher, invmask0);
				inv1 = _mm_and_si128(cipher, invmask1);
				inv2 = _mm_and_si128(cipher, invmask2);
				inv3 = _mm_and_si128(cipher, invmask3);

				index0 = (unsigned)(_mm_extract_epi32(inv0, 0) ^ _mm_extract_epi32(inv0, 1) ^ _mm_extract_epi32(inv0, 2) ^ _mm_extract_epi32(inv0, 3));
				index1 = (unsigned)(_mm_extract_epi32(inv1, 0) ^ _mm_extract_epi32(inv1, 1) ^ _mm_extract_epi32(inv1, 2) ^ _mm_extract_epi32(inv1, 3));
				index2 = (unsigned)(_mm_extract_epi32(inv2, 0) ^ _mm_extract_epi32(inv2, 1) ^ _mm_extract_epi32(inv2, 2) ^ _mm_extract_epi32(inv2, 3));
				index3 = (unsigned)(_mm_extract_epi32(inv3, 0) ^ _mm_extract_epi32(inv3, 1) ^ _mm_extract_epi32(inv3, 2) ^ _mm_extract_epi32(inv3, 3));

				Hash0[index0].push_back(plain1);
				Hash1[index1].push_back(plain1);
				Hash2[index2].push_back(plain1);
				Hash3[index3].push_back(plain1);

			}

			for (const auto& kv : Hash0) {
				const std::vector<__m128i>& vec0 = kv.second; 

				if (vec0.size() >= 2) {
					for (size_t b = 0; b < vec0.size(); ++b) {
						for (size_t c = b + 1; c < vec0.size(); ++c) {
							if (is_equal(vec0[b], vec0[c]) == 0)
							{
								memcpy(&Tab0[2 * m0], &vec0[b], sizeof(__m128i));
								memcpy(&Tab0[2 * m0 + 1], &vec0[c], sizeof(__m128i));
								m0++;
							}

						}
					}
				}
			}
			for (const auto& kv : Hash1) {
				const std::vector<__m128i>& vec1 = kv.second; 

				if (vec1.size() >= 2) {
					for (size_t b = 0; b < vec1.size(); ++b) {
						for (size_t c = b + 1; c < vec1.size(); ++c) {
							if (is_equal(vec1[b], vec1[c]) == 0)

							{
								memcpy(&Tab1[2 * m1], &vec1[b], sizeof(__m128i));
								memcpy(&Tab1[2 * m1 + 1], &vec1[c], sizeof(__m128i));
								m1++;
							}

						}
					}
				}
			}
			for (const auto& kv : Hash2) {
				const std::vector<__m128i>& vec2 = kv.second; 

				if (vec2.size() >= 2) {
					for (size_t b = 0; b < vec2.size(); ++b) {
						for (size_t c = b + 1; c < vec2.size(); ++c) {
							if (is_equal(vec2[b], vec2[c]) == 0)
							{
								memcpy(&Tab2[2 * m2], &vec2[b], sizeof(__m128i));
								memcpy(&Tab2[2 * m2 + 1], &vec2[c], sizeof(__m128i));
								m2++;
							}

						}
					}
				}
			}
			for (const auto& kv : Hash3) {
				const std::vector<__m128i>& vec3 = kv.second; 

				if (vec3.size() >= 2) {
					for (size_t b = 0; b < vec3.size(); ++b) {
						for (size_t c = b + 1; c < vec3.size(); ++c) {
							if (is_equal(vec3[b], vec3[c]) == 0)
							{
								memcpy(&Tab3[2 * m3], &vec3[b], sizeof(__m128i));
								memcpy(&Tab3[2 * m3 + 1], &vec3[c], sizeof(__m128i));
								m3++;
							}

						}
					}
				}
			}



			for (auto& kv : Hash0) {
				kv.second.clear();  
			}
			Hash0.clear();         
			for (auto& kv : Hash1) {
				kv.second.clear(); 
			}
			Hash1.clear();          
			for (auto& kv : Hash2) {
				kv.second.clear();  
			}
			Hash2.clear();          
			for (auto& kv : Hash3) {
				kv.second.clear(); 
			}
			Hash3.clear();        


			//printf("m0=%d,m1=%d,m2=%d,m3=%d\n", m0, m1, m2, m3);

			for (b = 0; b < m0; b++)
			{
				memcpy(&plain1, &Tab0[2 * b], sizeof(__m128i));
				memcpy(&plain2, &Tab0[2 * b + 1], sizeof(__m128i));
				for (j1 = 1; j1 < N; j1++)  //2^16
				{
					dif = _mm_set_epi16((j1 & 0xff) << 8, 0, 0, 0, 0, j1 & 0xff00, 0, 0);
					plain3 = _mm_xor_si128(plain1, dif);
					plain4 = _mm_xor_si128(plain2, dif);
					if (is_equal(plain3, plain4) == 0)
					{
						encrypt(plain3, key, &cipher3);
						encrypt(plain4, key, &cipher4);
						if (judge(cipher3, cipher4, 0) == 1)
						{
							con[i] = 1;
							break;
						}
					}
				}
				if (con[i] > 0)
				{
					break;
				}
			}



			for (b = 0; b < m1; b++)
			{
				memcpy(&plain1, &Tab1[2 * b], sizeof(__m128i));
				memcpy(&plain2, &Tab1[2 * b + 1], sizeof(__m128i));
				for (j1 = 1; j1 < N; j1++)  //2^16
				{
					dif = _mm_set_epi16((j1 & 0xff) << 8, 0, 0, 0, 0, j1 & 0xff00, 0, 0);
					plain3 = _mm_xor_si128(plain1, dif);
					plain4 = _mm_xor_si128(plain2, dif);
					if (is_equal(plain3, plain4) == 0)
					{
						encrypt(plain3, key, &cipher3);
						encrypt(plain4, key, &cipher4);
						if (judge(cipher3, cipher4, 1) == 1)
						{
							con[i] = 1;
							break;
						}
					}
				}
				if (con[i] > 0)
				{
					break;
				}
			}



			for (b = 0; b < m2; b++)
			{
				memcpy(&plain1, &Tab2[2 * b], sizeof(__m128i));
				memcpy(&plain2, &Tab2[2 * b + 1], sizeof(__m128i));
				for (j1 = 1; j1 < N; j1++)  //2^16
				{
					dif = _mm_set_epi16((j1 & 0xff) << 8, 0, 0, 0, 0, j1 & 0xff00, 0, 0);
					plain3 = _mm_xor_si128(plain1, dif);
					plain4 = _mm_xor_si128(plain2, dif);
					if (is_equal(plain3, plain4) == 0)
					{
						encrypt(plain3, key, &cipher3);
						encrypt(plain4, key, &cipher4);
						if (judge(cipher3, cipher4, 2) == 1)
						{
							con[i] = 1;
							break;
						}
					}
				}
				if (con[i] > 0)
				{
					break;
				}
			}



			for (b = 0; b < m3; b++)
			{
				memcpy(&plain1, &Tab3[2 * b], sizeof(__m128i));
				memcpy(&plain2, &Tab3[2 * b + 1], sizeof(__m128i));
				for (j1 = 1; j1 < N; j1++)  //2^16
				{
					dif = _mm_set_epi16((j1 & 0xff) << 8, 0, 0, 0, 0, j1 & 0xff00, 0, 0);
					plain3 = _mm_xor_si128(plain1, dif);
					plain4 = _mm_xor_si128(plain2, dif);
					if (is_equal(plain3, plain4) == 0)
					{
						encrypt(plain3, key, &cipher3);
						encrypt(plain4, key, &cipher4);
						if (judge(cipher3, cipher4, 3) == 1)
						{
							con[i] = 1;
							break;
						}
					}
				}
				if (con[i] > 0)
				{
					break;
				}
			}
			if (con[i] > 0)
			{
				break;
			}
		}


		if (con[i] > 0)
		{
			count++;
		}
	}
	printf("there are %d experiments in total, where %d results is 5-round AES\n\n", test_number, count);
	endtime = time(NULL);
	printf("time=%f\n", difftime(endtime, starttime));

}





		