

#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define VERBOSE

#define tabsize(t) (sizeof(t)/sizeof((t)[0]))

/***********************************/
/* T-tables based AES              */
/***********************************/


typedef uint16_t word;
#define MASK 0xf
#define M0 0xf000
#define M1 0x0f00
#define M2 0x00f0
#define M3 0x000f

#define N0 0x0fff
#define N1 0xf0ff
#define N2 0xff0f
#define N3 0xfff0
#include "tables4.h"



typedef word state[4];


#define M0 0xf000
#define M1 0x0f00
#define M2 0x00f0
#define M3 0x000f

#define N0 0x0fff
#define N1 0xf0ff
#define N2 0xff0f
#define N3 0xfff0



//result of MC SB and its inverse
word Table[65536];
word Tableinv[65536];

//result of SB and its inverse
word Tab[65536];
word Tabinv[65536];



void PreTable()
{
    uint8_t k0, k1, k2, k3;
    for (k0 = 0; k0 < 16; k0++)
        for (k1 = 0; k1 < 16; k1++)
            for (k2 = 0; k2 < 16; k2++)
                for (k3 = 0; k3 < 16; k3++)
                {
                    Table[(k0 << 12) ^ (k1 << 8) ^ (k2 << 4) ^ k3] = Te0[k0] ^ Te1[k1] ^ Te2[k2] ^ Te3[k3];
                    Tableinv[(k0 << 12) ^ (k1 << 8) ^ (k2 << 4) ^ k3] = Td0[k0] ^ Td1[k1] ^ Td2[k2] ^ Td3[k3];
                    Tab[(k0 << 12) ^ (k1 << 8) ^ (k2 << 4) ^ k3] = (Te4[k0] & M0) ^ (Te4[k1] & M1) ^ (Te4[k2] & M2) ^ (Te4[k3] & M3);
                    Tabinv[(k0 << 12) ^ (k1 << 8) ^ (k2 << 4) ^ k3] = (Td4[k0] & M0) ^ (Td4[k1] & M1) ^ (Td4[k2] & M2) ^ (Td4[k3] & M3);
                }
}



void Round1(state x,  state y) {
    word index0, index1, index2, index3;
    //SR
    index0 = (x[0] & M0) ^ (x[1] & M1) ^ (x[2] & M2) ^ (x[3] & M3);
    index1 = (x[1] & M0) ^ (x[2] & M1) ^ (x[3] & M2) ^ (x[0] & M3);
    index2 = (x[2] & M0) ^ (x[3] & M1) ^ (x[0] & M2) ^ (x[1] & M3);
    index3 = (x[3] & M0) ^ (x[0] & M1) ^ (x[1] & M2) ^ (x[2] & M3);
    //MC SB
    y[0] = Table[index0];
    y[1] = Table[index1];
    y[2] = Table[index2];
    y[3] = Table[index3];
}




void OneRoundEncrypt(state x1, state x2, state key, state y) {
    state z1, z2, y1, y2;
    memcpy(z1, x1, sizeof(state));
    z1[0] ^= key[0];
    z1[1] ^= key[1];
    z1[2] ^= key[2];
    z1[3] ^= key[3];
    Round1(z1, y1);

    memcpy(z2, x2, sizeof(state));
    z2[0] ^= key[0];
    z2[1] ^= key[1];
    z2[2] ^= key[2];
    z2[3] ^= key[3];
    Round1(z2, y2);

    y[0] = y1[0] ^ y2[0];
    y[1] = y1[1] ^ y2[1];
    y[2] = y1[2] ^ y2[2];
    y[3] = y1[3] ^ y2[3];
}


uint8_t dia_weight(state x)
{
    state z;
    memcpy(z, x, sizeof(state));
    return
        !!((z[0] & M0) ^ (z[1] & M1) ^ (z[2] & M2) ^ (z[3] & M3)) +
        !!((z[1] & M0) ^ (z[2] & M1) ^ (z[3] & M2) ^ (z[0] & M3)) +
        !!((z[2] & M0) ^ (z[3] & M1) ^ (z[0] & M2) ^ (z[1] & M3)) +
        !!((z[3] & M0) ^ (z[0] & M1) ^ (z[1] & M2) ^ (z[2] & M3));
}

char exchange(state x, state y, uint8_t i, state z, state w)  //exchange the i-th diagonal of x and y to obtain z and w
{
    if (i == 0) {
        z[0] = (x[0] & N0) ^ (y[0] & M0);
        z[1] = (x[1] & N1) ^ (y[1] & M1);
        z[2] = (x[2] & N2) ^ (y[2] & M2);
        z[3] = (x[3] & N3) ^ (y[3] & M3);

        w[0] = (y[0] & N0) ^ (x[0] & M0);
        w[1] = (y[1] & N1) ^ (x[1] & M1);
        w[2] = (y[2] & N2) ^ (x[2] & M2);
        w[3] = (y[3] & N3) ^ (x[3] & M3);
        return 1;
    }
    if (i == 1) {
        z[0] = (x[0] & N3) ^ (y[0] & M3);
        z[1] = (x[1] & N0) ^ (y[1] & M0);
        z[2] = (x[2] & N1) ^ (y[2] & M1);
        z[3] = (x[3] & N2) ^ (y[3] & M2);

        w[0] = (y[0] & N3) ^ (x[0] & M3);
        w[1] = (y[1] & N0) ^ (x[1] & M0);
        w[2] = (y[2] & N1) ^ (x[2] & M1);
        w[3] = (y[3] & N2) ^ (x[3] & M2);
        return 1;
    }
    if (i == 2) {
        z[0] = (x[0] & N2) ^ (y[0] & M2);
        z[1] = (x[1] & N3) ^ (y[1] & M3);
        z[2] = (x[2] & N0) ^ (y[2] & M0);
        z[3] = (x[3] & N1) ^ (y[3] & M1);

        w[0] = (y[0] & N2) ^ (x[0] & M2);
        w[1] = (y[1] & N3) ^ (x[1] & M3);
        w[2] = (y[2] & N0) ^ (x[2] & M0);
        w[3] = (y[3] & N1) ^ (x[3] & M1);
        return 1;
    }
    if (i == 3) {
        z[0] = (x[0] & N1) ^ (y[0] & M1);
        z[1] = (x[1] & N2) ^ (y[1] & M2);
        z[2] = (x[2] & N3) ^ (y[2] & M3);
        z[3] = (x[3] & N0) ^ (y[3] & M0);

        w[0] = (y[0] & N1) ^ (x[0] & M1);
        w[1] = (y[1] & N2) ^ (x[1] & M2);
        w[2] = (y[2] & N3) ^ (x[2] & M3);
        w[3] = (y[3] & N0) ^ (x[3] & M0);
        return 1;
    }
}

void print_state(state x) {
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
            printf("%01x", (x[j] >> (4 * (3 - i))) & 0xf);
        printf("\n");
    }

    printf("\n");
}

char dia_equal(state x, state y, uint8_t i)   //whether the i-th diagonals of x and y are equal
{

    uint16_t z, w;
    if (i == 0) {
        z = (x[0] & M0) ^ (x[1] & M1) ^ (x[2] & M2) ^ (x[3] & M3);
        w = (y[0] & M0) ^ (y[1] & M1) ^ (y[2] & M2) ^ (y[3] & M3);
        if (z == w)
            return 1;
        else return 0;
    }
    if (i == 1) {
        z = (x[1] & M0) ^ (x[2] & M1) ^ (x[3] & M2) ^ (x[0] & M3);
        w = (y[1] & M0) ^ (y[2] & M1) ^ (y[3] & M2) ^ (y[0] & M3);
        if (z == w)
            return 1;
        else return 0;
    }
    if (i == 2) {
        z = (x[2] & M0) ^ (x[3] & M1) ^ (x[0] & M2) ^ (x[1] & M3);
        w = (y[2] & M0) ^ (y[3] & M1) ^ (y[0] & M2) ^ (y[1] & M3);
        if (z == w)
            return 1;
        else return 0;
    }
    if (i == 3) {
        z = (x[3] & M0) ^ (x[0] & M1) ^ (x[1] & M2) ^ (x[2] & M3);
        w = (y[3] & M0) ^ (y[0] & M1) ^ (y[1] & M2) ^ (y[2] & M3);
        if (z == w)
            return 1;
        else return 0;
    }
}