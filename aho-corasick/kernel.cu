
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//definitions
#define NUM_BLOCKS *
#define BLOCK_DIM *
#define S_CHUNK *
#define S_THREAD *
#define S_MEMSIZE *

//function definitions
void input();
void preprocessing();
void formatTries();
void allocateData();
void computeOnDevice();
__device__ void AhoCorasickKernel();
void calculateResult();
void computeGold();
void compareWithGold();
void freeMemory();


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
    input();

    // Execute  the  preprocessing  phase  of  the  algorithms
    preprocessing();

    // Where  relevant  represent  t r i e s  using  arrays
    formatTries();

    // Allocate  one−dimensional and two−dimensional  arrays  in  the  global  memoryof  the  device  using  the  cudaMalloc() and cudaMallocPitch()functions  r e s p e c t i v e l y
    allocateData();

    computeOnDevice();
    
    calculateResult();

    computeGold();

    compareWithGold();

    freeMemory();
}

void input()
{

}

void preprocessing()
{

}

void formatTries()
{

}

void allocateData()
{

}

void computeOnDevice()
{

}

__device__ void AhoCorasickKernel()
{

}

void calculateResult()
{

}

void computeGold()
{
    // use aho-corasick algorithm from https://github.com/morenice/ahocorasick
}

void computeGold()
{

}

void compareWithGold()
{

}

void freeMemory()
{

}