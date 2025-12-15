#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>


#define NUM_STATES_EXP 32          // 2^32 states
#define NUM_STATES (1ULL << NUM_STATES_EXP)
#define NUM_DELTAS 256             // differences 0-255
#define CHUNK_SIZE (1 << 20)       
#define TEST_ROUNDS 100              // test numbers


static __m128i round_key1, round_key2;


static inline __m128i aes_two_rounds(__m128i state, __m128i key1, __m128i key2) {
    __m128i middle = _mm_aesenc_si128(state, key1);
    return _mm_aesenclast_si128(middle, key2);
}


static inline __m128i create_diagonal_state(uint32_t diag_value, uint8_t fixed_bytes[12]) {
 
    uint8_t b0 = (diag_value >> 24) & 0xFF;
    uint8_t b1 = (diag_value >> 16) & 0xFF;
    uint8_t b2 = (diag_value >> 8) & 0xFF;
    uint8_t b3 = diag_value & 0xFF;

    uint8_t state_bytes[16] = {
        b0,        fixed_bytes[0],  fixed_bytes[1],  fixed_bytes[2],
        fixed_bytes[3],  b1,        fixed_bytes[4],  fixed_bytes[5],
        fixed_bytes[6],  fixed_bytes[7],  b2,        fixed_bytes[8],
        fixed_bytes[9],  fixed_bytes[10], fixed_bytes[11], b3
    };

    return _mm_loadu_si128((const __m128i*)state_bytes);
}


static inline void extract_target_diff(__m128i delta_state, uint8_t target_delta[4]) {
    alignas(16) uint8_t delta_bytes[16];
    _mm_store_si128((__m128i*)delta_bytes, delta_state);

    target_delta[0] = delta_bytes[0];   // (0,0)
    target_delta[1] = delta_bytes[13];  // (1,3)
    target_delta[2] = delta_bytes[10];  // (2,2)
    target_delta[3] = delta_bytes[7];   // (3,1)
}



int perform_single_test(int round_idx) {
    printf("==============================================================\n");
    printf("                           Test %d                            \n", round_idx + 1);
    printf("==============================================================\n\n");

  
    uint8_t key1_bytes[16], key2_bytes[16];
    for (int i = 0; i < 16; i++) {
        key1_bytes[i] = rand() & 0xFF;
        key2_bytes[i] = 0;
    }

    round_key1 = _mm_loadu_si128((const __m128i*)key1_bytes);
    round_key2 = _mm_loadu_si128((const __m128i*)key2_bytes);

    printf("The Round Key: ");
    for (int i = 0; i < 16; i++) printf("%02x", key1_bytes[i]);
    printf("\n");
    

    uint8_t fixed_bytes[12];
    for (int i = 0; i < 12; i++) {
        fixed_bytes[i] = rand() & 0xFF;
    }


    int max_threads = omp_get_max_threads();
    printf("Detected CPU cores: %d\n", max_threads);
    printf("Parallel threads set to: %d\n\n", max_threads);

    size_t counts_size = (size_t)max_threads * 4 * 256 * sizeof(uint64_t);
    size_t totals_size = (size_t)max_threads * sizeof(uint64_t);
    uint64_t* counts_buffer = (uint64_t*)calloc(max_threads * 4 * 256, sizeof(uint64_t));
    uint64_t* thread_totals = (uint64_t*)calloc(max_threads, sizeof(uint64_t));

    if (!counts_buffer || !thread_totals) {
        printf("Memory allocation failed!\n");
        if (counts_buffer) free(counts_buffer);
        if (thread_totals) free(thread_totals);
        return -1;
    }

    time_t start_time = time(NULL);
  


#pragma omp parallel num_threads(max_threads)
    {
        int thread_id = omp_get_thread_num();
        uint64_t local_total = 0;
        uint64_t* my_counts = counts_buffer + thread_id * 4 * 256;

        if (thread_id >= max_threads) {
            printf("Warning: Thread ID %d is out of range\n", thread_id);
        }

#pragma omp for schedule(dynamic, CHUNK_SIZE)
        for (uint64_t state_idx = 0; state_idx < NUM_STATES; state_idx++) {
            uint32_t diag_value = (uint32_t)state_idx;
            __m128i plain1 = create_diagonal_state(diag_value, fixed_bytes);

            for (int delta_int = 0; delta_int <= 255; delta_int++) {
                uint8_t delta = (uint8_t)delta_int;
                __m128i delta_mask = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, delta);
                __m128i plain2 = _mm_xor_si128(plain1, delta_mask);

                __m128i cipher1 = aes_two_rounds(plain1, round_key1, round_key2);
                __m128i cipher2 = aes_two_rounds(plain2, round_key1, round_key2);
                __m128i cipher_delta = _mm_xor_si128(cipher1, cipher2);

                uint8_t target_delta[4];
                extract_target_diff(cipher_delta, target_delta);

                my_counts[0 * 256 + target_delta[0]]++;
                my_counts[1 * 256 + target_delta[1]]++;
                my_counts[2 * 256 + target_delta[2]]++;
                my_counts[3 * 256 + target_delta[3]]++;

                local_total++;
            }
        }
        thread_totals[thread_id] = local_total;
    }


    time_t end_time = time(NULL);
    double total_elapsed = difftime(end_time, start_time);
    printf("Total time for Test %d: %.0f seconds (%.2f hours)\n", round_idx + 1, 
        total_elapsed, total_elapsed / 3600.0);


    uint64_t global_counts[4][256] = { 0 };
    uint64_t total_tests = 0;
    for (int t = 0; t < max_threads; t++) {
        uint64_t* thread_counts = counts_buffer + t * 4 * 256;
        for (int pos = 0; pos < 4; pos++) {
            uint64_t* pos_counts = thread_counts + pos * 256;
            for (int val = 0; val < 256; val++) {
                global_counts[pos][val] += pos_counts[val];
            }
        }
        total_tests += thread_totals[t];
    }


    const char* pos_names[4] = { "0", "13", "10", "7" };
    int round_uniform = 1; 

    for (int pos = 0; pos < 4; pos++) {
        printf("In the %s-th byte: ", pos_names[pos]);
        //printf("------------------------------------------------------------\n");

        double expected = (double)total_tests / 256.0;
        char pos_flag = 0;

        //printf("Expected frequency per difference: %.1f\n", expected);
        for (int val = 0; val < 256; val++) {
            uint64_t count = global_counts[pos][val];
            if (count != expected) {
                printf("difference 0x%02x occurs %llu times instead of %.1f\n",
                    val, (unsigned long long)count, expected);
                pos_flag = 1;
                round_uniform = 0;
            }
        }
        if (pos_flag == 0) {
            printf("each difference occurs %.1f times\n", expected);
        }
        printf("\n");
    }

    if (round_uniform) {
        printf("Conclusion of test %d: Differences in the target bytes are uniform. \n", round_idx + 1);
    }
    else {
        printf("Conclusion of test %d: Non-uniform distribution. \n", round_idx + 1);
    }


    free(counts_buffer);
    free(thread_totals);

    return round_uniform  ? 0 : -1;
}




int main() {
    srand((unsigned int)time(NULL));


    printf("The Total Number of Tests：%d\n", TEST_ROUNDS);

    int success_rounds = 0;
    for (int i = 0; i < TEST_ROUNDS; i++) {
        if (perform_single_test(i) == 0) {
            success_rounds++;
        }
        printf("\n\n");
    }

    printf("==============================================================\n");
    printf("                        Test Summary                          \n");
    printf("==============================================================\n");
    printf("Total number of tests: %d\n", TEST_ROUNDS);
    printf("Number of tests with uniform distribution: %d\n", success_rounds);
    printf("Pass probability: %.2f%%\n", (double)success_rounds / TEST_ROUNDS * 100.0);

}
