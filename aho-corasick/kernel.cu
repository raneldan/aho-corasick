
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "include/hoDll.h"

//definitions
#define NUM_BLOCKS 1
#define BLOCK_DIM 1024
#define N 1000 // size of search string. devisable by NUM_BLOCKS and by BLOCK_DIM
#define M 10 // count of patterns
#define S_CHUNK (N / NUM_BLOCKS) // amount of characters in a chunk
#define S_THREAD (N / (NUM_BLOCKS * BLOCK_DIM) + M - 1) // amount of characters proccessed by a thread
#define S_MEMSIZE *

//function definitions
void preprocessing();
void formatTries();
void allocateData();
void computeOnDevice(char*, char*, const size_t, const size_t);
__device__ void AhoCorasickKernel(char*, char*, unsigned int*);
void calculateResult();
void computeGold();
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

int main()
{
    const size_t PATTERN_LENGTH = 10;
    char searchphase[N];
    char pattern[PATTERN_LENGTH];
    
    // input
    FILE* fp;

    fp = fopen("Input.txt", "r");
    fgets(searchphase, N, fp);
    fclose(fp);

    // Execute  the  preprocessing  phase  of  the  algorithms
    preprocessing();

    // Where  relevant  represent  t r i e s  using  arrays
    formatTries();

    // Allocate  one−dimensional and two−dimensional  arrays  in  the  global  memoryof  the  device  using  the  cudaMalloc() and cudaMallocPitch()functions  r e s p e c t i v e l y
    allocateData();

    //computeOnDevice();
    
    calculateResult();

    computeGold();

    compareWithGold();

    // free memory
}

void preprocessing()
{
    int i;

}

void formatTries()
{
    int i;
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

void computeGold()
{
    // use aho-corasick algorithm from https://github.com/morenice/ahocorasick
    int i;
}

void compareWithGold()
{
    int i;
}