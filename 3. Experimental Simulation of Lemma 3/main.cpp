#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include "PreComputation.c"

#define test_number (UINT64_C(50) << 32)   //50*2^32


int main() {
    state plain1, plain2, plain3, plain4, plain5,plain6;
    state cipher_diff1, cipher_diff2;
    state masterkey;
    uint64_t i;

    uint64_t* random;
    random = (uint64_t*)malloc(sizeof(uint64_t));
    _rdrand64_step(random);
    memcpy(masterkey, random, sizeof(state));

    uint32_t counter = 0;
    PreTable();
    time_t starttime, endtime;
    starttime = time(NULL);

    for (i = 0; i < test_number; i++)
    {
        _rdrand64_step(random);
        memcpy(plain1, random, sizeof(state));

        _rdrand64_step(random);
        memcpy(plain2, random, sizeof(state));

        //print_state(plain1);
        //print_state(plain2);
        exchange(plain1, plain2, 0, plain3, plain4);
        //exchange(plain3, plain4, 1, plain5, plain6);
        //print_state(plain3);
        //print_state(plain4);

        OneRoundEncrypt(plain1, plain2, masterkey, cipher_diff1);
        if (dia_weight(cipher_diff1) == 4) {
            OneRoundEncrypt(plain1, plain3, masterkey, cipher_diff2);
            //OneRoundEncrypt(plain1, plain5, masterkey, cipher_diff2);
            if (dia_weight(cipher_diff2) == 2)
            {
                if (dia_equal(cipher_diff1, cipher_diff2, 0) == 1 && dia_equal(cipher_diff1, cipher_diff2, 2) == 1)
                 //if (dia_equal(cipher_diff1, cipher_diff2, 1) == 1 && dia_equal(cipher_diff1, cipher_diff2, 3) == 1)
                {
                   // print_state(cipher_diff1);
                   // print_state(cipher_diff2);
                    counter++;
                }

            }
        }        
    }
    printf("there are %lld experiments in total, where %d results satisfy the requirement\n", test_number, counter);
    endtime = time(NULL);
    printf("time=%f\n", difftime(endtime, starttime));
    return counter;
}


