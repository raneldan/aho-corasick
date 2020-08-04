
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "include/hoDll.h"

//definitions
#define NUM_BLOCKS 1
#define BLOCK_DIM 1024
#define N 1000000 // size of search string. devisable by NUM_BLOCKS and by BLOCK_DIM
#define M 10 // count of patterns
#define S_CHUNK (N / NUM_BLOCKS) // amount of characters in a chunk
#define S_THREAD (N / (NUM_BLOCKS * BLOCK_DIM) + M - 1) // amount of characters proccessed by a thread
#define S_MEMSIZE *

#define trie_state uint32_t
#define SIGMA_SIZE 52 // small case and big case english letters.

//define list of patterns
const char* P[M] = {
    "consectetur",
    "est",
    "egestasAliquam",
    "elementum",
    "ultricies",
    "vehicula",
    "tortor",
    "inauditum",
    "inquam" ,
    "aut"
};

// calculate maximum number of states, should be the sum of lengths of patterns.
const size_t MAX_NUM_OF_STATES = 78;


//function definitions
void preprocessing();
void allocateData();
void computeOnDevice(char*, char*, const size_t, const size_t);
__device__ void AhoCorasickKernel(char*, char*, unsigned int*);
void calculateResult();
void computeGold(const char*, const char**);
void compareWithGold();


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/*void callback_match_pos(void* arg, struct aho_match_t* m)
{
    char* text = (char*)arg;

    printf("match text:");
    for (unsigned int i = m->pos; i < (m->pos + m->len); i++)
    {
        printf("%c", text[i]);
    }

    printf(" (match id: %d position: %llu length: %d)\n", m->id, m->pos, m->len);
}*/

int main()
{
    char searchphase[N];
    char pattern[M];
    
    // input
    FILE* fp;

    fp = fopen("Input.txt", "r");
    fgets(searchphase, N, fp);
    fclose(fp);

    // Execute  the  preprocessing  phase  of  the  algorithm
    preprocessing();

    // Where  relevant  represent  tries  using  arrays
    
    // Allocate  one−dimensional and two−dimensional  arrays  in  the  global  memoryof  the  device  using  the  cudaMalloc() and cudaMallocPitch()functions  r e s p e c t i v e l y
    //allocateData();

    //computeOnDevice();
    
    //calculateResult();

    computeGold(searchphase, P);

    //compareWithGold();

    // free memory
    //aho_destroy(&aho);
}

void preprocessing()
{
    // implementation inspired by https://www.geeksforgeeks.org/aho-corasick-algorithm-pattern-searching/
    trie_state State_transition[MAX_NUM_OF_STATES][SIGMA_SIZE]; // corespondes to goto function. row = current state, col = character in Σ, value = next state.
    trie_state State_supply[MAX_NUM_OF_STATES]; //corespondes to supply function. col = current state, value = supply state + is it final state.
    trie_state State_output[MAX_NUM_OF_STATES]; //corespondes to output function. col = state, value = 1 if word ends at this state, 0 otherwise.

    memset(State_output, 0, sizeof State_output);
    memset(State_transition, UINT32_MAX, sizeof State_transition);

    int states = 1;

    for (int i = 0; i < M; ++i)
    {
        const char* word = P[i];
        trie_state currentState = 0;

        for (int j = 0; j < strlen(word); j++)
        {
            int ch = word[j] - 'a';
            if (State_transition[currentState][ch] == UINT32_MAX)
            {
                State_transition[currentState][ch] = states++;
            }

            currentState = State_transition[currentState][ch];
        }

        State_output[currentState] |= (1 << i);
    }

    for (int ch = 0; ch < SIGMA_SIZE; ++ch)
    {
        if (State_transition[0][ch] == UINT32_MAX)
            State_transition[0][ch] = 0;
    }

    memset(State_supply, UINT32_MAX, sizeof State_supply);
    queue<trie_state> q;

    for (int ch = 0; ch < SIGMA_SIZE; ++ch)
    {
        if (State_transition[0][ch] != 0)
        {
            State_supply[State_transition[0][ch]] = 0;
            q.push(State_transition[0][ch]);

            //TODO continue this
        }
    }
}

void allocateData()
{
    int i;
}

void computeOnDevice(char* h_searchphase, char* h_tries, const size_t phase_size, const size_t tries_size)
{
    size_t out_size = NUM_BLOCKS * BLOCK_DIM;

    unsigned int* h_out; // number of matches per thread
    char* d_searchphase;
    char* d_tries;
    unsigned int* d_out;

    h_out = static_cast<unsigned int*>(malloc(out_size));
    /*cudaMalloc(d_searchphase, phase_size);
    cudaMalloc(d_tries, tries_size);
    cudaMalloc(d_out, out_size);*/

    cudaMemcpy(d_searchphase, h_searchphase, phase_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tries, h_tries, tries_size, cudaMemcpyHostToDevice);

    //AhoCorasickKernel << NUM_BLOCKS, BLOCK_DIM >> (d_searchphase, d_tries, d_out);

    cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost);
}

__device__ void AhoCorasickKernel(char* d_searchphase, char* d_tries, unsigned int* d_out)
{
    size_t blockId = blockIdx.x;
    size_t threadId = threadIdx.x;
    
    // define start and stop auxilary variables.
    size_t start, stop; //used to indicate the input string positions where the search phase begins and ends respectively
    unsigned int overlap = M - 1; // To ensure the correctness of the results, m−1 overlapping characters are used per thread
    start = (blockId * N) / NUM_BLOCKS + (N * threadId) / (NUM_BLOCKS * blockDim.x);
    stop = start + N / (NUM_BLOCKS * blockDim.x) + overlap;

}

void calculateResult()
{
    int i;
}

void computeGold(const char* target, const char* P[M])
{
    clock_t start, end;

    struct ahocorasick aho;
    int id[M] = { 0 };
    //char* target = "Lorem ipsum dolor sit amet, consectetur brown elit. Proin vehicula brown egestas. Aliquam a dui tincidunt, elementum sapien in, ultricies lacus. Phasellus congue, sapien nec";
    aho_init(&aho);

    id[0] = aho_add_match_text(&aho, P[0], strlen(P[0]));
    id[1] = aho_add_match_text(&aho, P[1], strlen(P[1]));
    id[2] = aho_add_match_text(&aho, P[2], strlen(P[2]));
    id[3] = aho_add_match_text(&aho, P[3], strlen(P[3]));
    id[4] = aho_add_match_text(&aho, P[4], strlen(P[4]));
    id[5] = aho_add_match_text(&aho, P[5], strlen(P[5]));
    id[6] = aho_add_match_text(&aho, P[6], strlen(P[6]));
    id[7] = aho_add_match_text(&aho, P[7], strlen(P[7]));
    id[8] = aho_add_match_text(&aho, P[8], strlen(P[8]));
    id[9] = aho_add_match_text(&aho, P[9], strlen(P[9]));

    aho_create_trie(&aho);
    //aho_register_match_callback(&aho, callback_match_pos, (void*)searchphase);

    // use aho-corasick algorithm from https://github.com/morenice/ahocorasick
    start = clock();
    printf("total match CPU:%u\n", aho_findtext(&aho, target, strlen(target)));
    end = clock();

    double milliseconds = ((double)(end - start)) / double(CLOCKS_PER_SEC);
    printf("Execution Time Gold CPU (sec): %lf\n", milliseconds);
}

void compareWithGold()
{
    int i;
}