#ifndef RAND_H
#include <cstdlib>
#include <cstdint>
#include "immintrin.h"
#include <assert.h>

#define MY_RAND_MAX 0x7fffffff

void init_rand_state();
const int CACHE_LINE_SIZE = 64;
const int MAX_NUM_THREADS = 64;

uint32_t xorshift128plus(uint64_t* state);


extern uint64_t rand_state[2];
#define RAND() xorshift128plus(&rand_state[0])


#define RAND_H
#endif