
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

#include "simpleQ.h"
#include "include/hoDll.h"

//options
//#define SIMPLE_EXAMPLE
//#define DEBUG

//optimizations
//#define CAST_TO_INT
#define GRANUALITY 3

//hardware
#define NUM_OF_SM 6
#define S_MEM_PER_SM 48000
#define NUM_SHARED_MEMORY_BANKS 32

//definitions
#ifdef SIMPLE_EXAMPLE
#define NUM_BLOCKS 1
#define BLOCK_DIM 2
#define N 8 // size of search string. devisable by NUM_BLOCKS and by BLOCK_DIM
#else
#define NUM_BLOCKS_FACTOR 10
#define BLOCK_DIM_FACTOR 10
#define N_FACTOR 30
#define NUM_BLOCKS (NUM_OF_SM * NUM_BLOCKS_FACTOR)
#define BLOCK_DIM (8 * BLOCK_DIM_FACTOR) // NUM_BLOCKS * BLOCK_DIM should be devisable by 16, so BLOCK_DIM should be a factor of 8 for NUM_OF_SM=6.
#define N (NUM_BLOCKS * BLOCK_DIM * N_FACTOR) // size of search string. devisable by NUM_BLOCKS and by BLOCK_DIM
#endif

#define trie_state uint32_t

#define SIGMA_SIZE ('z' - 'a' + 1) // small case english letters.

//define list of patterns
#ifdef SIMPLE_EXAMPLE
#define NUM_OF_PATTERNS 4 // count of patterns
#define LENGTH_OF_PATTERN 4 // const len of each pattern
const char* P[NUM_OF_PATTERNS] = {
    "hehe",
    "shee",
    "hiss",
    "hers"
};
#else
#define NUM_OF_PATTERNS 10 // count of patterns
#define LENGTH_OF_PATTERN 6
const char* P[NUM_OF_PATTERNS] = {
    "consec",
    "estabc",
    "egesta",
    "elemen",
    "ultric",
    "vehicu",
    "tortor",
    "inaudi",
    "inquam",
    "autabc"
};
#endif //SIMPLE_EXAMPLE

// calculate maximum number of states, should be the sum of lengths of patterns.
#define MAX_NUM_OF_STATES (NUM_OF_PATTERNS * LENGTH_OF_PATTERN)

// kernel indecies definitons
#define THREAD_WORK_LENGTH ((N * GRANUALITY) / (NUM_BLOCKS * BLOCK_DIM))
#define THREAD_START_STEP ((N * GRANUALITY) / (BLOCK_DIM * NUM_BLOCKS))
#define BLOCK_START_STEP (N / NUM_BLOCKS)
#define OVERLAP (LENGTH_OF_PATTERN - 1) // To ensure the correctness of the results, m−1 overlapping characters are used per thread

// queue defenitions for calculation of supply function
typedef	struct _trie_state_queue_entry {
    SIMPLEQ_ENTRY(_trie_state_queue_entry) entries;
    trie_state state;
};
SIMPLEQ_HEAD(queue_head_type, _trie_state_queue_entry) head;

#define S_CHUNK (N / NUM_BLOCKS) // amount of characters in a chunk
#define S_THREAD (N / (NUM_BLOCKS * BLOCK_DIM) + LENGTH_OF_PATTERN - 1) // amount of characters proccessed by a thread
#define S_MEMSIZE (S_MEM_PER_SM * NUM_OF_SM / NUM_BLOCKS) // 48KB of shared memory per SM.


#if S_THREAD > 128
// when reading 128 bytes per transaction, if more then 128 bytes are needed, memory will coalese.
#pragma message("Warning: S_THREAD bigger then 128, memory will coalese.")
#endif

//function definitions
int preprocessing();
unsigned int computeOnDevice(char*);
__global__ void AhoCorasickKernel(char const* const, trie_state const* const, const size_t, trie_state const* const, trie_state const* const, unsigned int* const);
__device__ trie_state findNextState(trie_state const* const, const size_t, trie_state const* const, const trie_state, const char);
trie_state findNextStateHost(trie_state const* const, const size_t, trie_state const* const, const trie_state, const char);
unsigned int computeGold(const char*, const char**);
void compareWithGold(const unsigned int, const unsigned int);


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


void callback_match_pos(void* arg, struct aho_match_t* m)
{
    char* text = (char*)arg;

    printf("\n\nmatch text in CPU:");
    for (unsigned int i = m->pos; i < (m->pos + m->len); i++)
    {
        printf("%c", text[i]);
    }

    printf(" (match id: %d position: %llu length: %d)\n", m->id, m->pos, m->len);
}

trie_state State_transition[MAX_NUM_OF_STATES][SIGMA_SIZE]; // corespondes to goto function. row = current state, col = character in Σ, value = next state.
trie_state State_supply[MAX_NUM_OF_STATES]; //corespondes to supply function. col = current state, value = supply state + is it final state.
trie_state State_output[MAX_NUM_OF_STATES]; //corespondes to output function. col = state, value = 1 if word ends at this state, 0 otherwise.


int main()
{
    char* searchphase;
    char searchphase_reference[N+1];
    char pattern[NUM_OF_PATTERNS];

    printf("NUM_BLOCKS = %d\nBLOCK_DIM = %d\nN= %d\n", NUM_BLOCKS, BLOCK_DIM, N);
    
    gpuErrchk(cudaMallocHost((void**)&searchphase, N + 1 * sizeof(char))); // +1 for null

#ifdef SIMPLE_EXAMPLE
    strcpy_s(searchphase, "ahhersis");
    strcpy_s(searchphase_reference, "ahhersis");
#else
    // input
    FILE* fp;

    fp = fopen("Input.txt", "r");
    fgets(searchphase, N + 1, fp);
    fclose(fp);

    fp = fopen("Input.txt", "r");
    fgets(searchphase_reference, N + 1, fp);
    fclose(fp);
#endif //SIMPLE_EXAMPLE

#ifdef DEBUG

    printf("\n\nsearchphase:\n 1 \t");
    for (int i = 0, j = 1, k = 0; i <= N; i++, j++)
    {
        printf("%c", searchphase[i]);

        if (j == LENGTH_OF_PATTERN - 1)
        {
            printf("|");

        }

        if ((i + 1) % (N / (NUM_BLOCKS * BLOCK_DIM)) == 0)
        {
            printf("\t");
            j = 0;
        }

        if ((i + 1) % (N / NUM_BLOCKS) == 0)
        {
            printf("\n %d \t", i);
        }
    }
    printf("\n\n");

    printf("P: { ");
    for (int i = 0; i < NUM_OF_PATTERNS; i++)
    {
        printf("%s, ", P[i]);
    }
    printf("}\n\n");
#endif

    // Execute  the  preprocessing  phase  of  the  algorithm, represent tries using arrays
    preprocessing();
    
    // Allocate  one−dimensional and two−dimensional  arrays  in  the  global  memoryof  the  device  using  the  cudaMalloc() and cudaMallocPitch()functions  r e s p e c t i v e l y

    unsigned int result = computeOnDevice(searchphase);

    printf("GPU RESULT: %d \n", result);
    
    unsigned int reference = computeGold(searchphase_reference, P);

    compareWithGold(result, reference);

    // free memory
    //aho_destroy(&aho);
}

int preprocessing()
{
    // implementation inspired by https://www.geeksforgeeks.org/aho-corasick-algorithm-pattern-searching/

    memset(State_output, 0, sizeof State_output);
    memset(State_transition, UINT32_MAX, sizeof State_transition);

    // build transition function
    int states = 1;

    for (int i = 0; i < NUM_OF_PATTERNS; ++i)
    {
        // calculate state transitions for each pattern
        const char* word = P[i];
        trie_state currentState = 0;

        for (int j = 0; j < strlen(word); j++)
        {
            // calculate state transition for each charcter in the pattern
            int ch = word[j] - 'a';
            if (State_transition[currentState][ch] == UINT32_MAX) {
                // if no transtion was yet set for this charcter in this state, then add a new state.
                State_transition[currentState][ch] = states++;
            }

            currentState = State_transition[currentState][ch];
        }

        // add word to output function
        State_output[currentState] |= (1 << i);
    }

    // all transitions from initial state that were not handled - return to initial state.
    for (int ch = 0; ch < SIGMA_SIZE; ++ch)
    {
        if (State_transition[0][ch] == UINT32_MAX) {
            State_transition[0][ch] = 0;
        }
    }

    // build supply function.
    //implemented using breadth first search using a queue.
    memset(State_supply, UINT32_MAX, sizeof State_supply);
    struct queue_head_type head = SIMPLEQ_HEAD_INITIALIZER(head);

    for (int ch = 0; ch < SIGMA_SIZE; ++ch)
    {
        // for every possible input
        if (State_transition[0][ch] != 0)
        {
            // for nodes in a pattern in depth 1
            _trie_state_queue_entry* tsqe;

            // all depth 1 nodes supply transition is to state 0
            // that is: failure after the first letter returns to the start state.
            State_supply[State_transition[0][ch]] = 0;

            // add node to queue for search later.
            tsqe = (_trie_state_queue_entry*)malloc(sizeof(struct _trie_state_queue_entry));
            tsqe->state = State_transition[0][ch];
            SIMPLEQ_INSERT_TAIL(&head, tsqe, entries);
        }
    }

    // now search over the states in the queue, breadth first
    while (!SIMPLEQ_EMPTY(&head))
    {
        // for each state in the queue, find supply function for all those
        // characters for which transition function is not defined.

        // remove state from the queue
        trie_state state = SIMPLEQ_FIRST(&head)->state;
        SIMPLEQ_REMOVE_HEAD(&head, entries);

        for (int ch = 0; ch < SIGMA_SIZE; ++ch)
        {
            // for each possible input
            if (State_transition[state][ch] != UINT32_MAX)
            {
                // if a state transition is defined for this input at this state

                // find currently defined supply for the state
                trie_state supply = State_supply[state];

                // find the deepest supply defined for the current state. 
                while (State_transition[supply][ch] == UINT32_MAX)
                    // while the supply state has a defined state transition set supply to it's own supply
                    supply = State_supply[supply];

                // finally set the supply to the transition for found state for the character.
                supply = State_transition[supply][ch];
                State_supply[State_transition[state][ch]] = supply;

                // Merge output values 
                State_output[State_transition[state][ch]] |= State_output[supply];

                // add node to queue for search later.
                _trie_state_queue_entry* tsqe;
                tsqe = (_trie_state_queue_entry*)malloc(sizeof(struct _trie_state_queue_entry));
                tsqe->state = State_transition[state][ch];
                SIMPLEQ_INSERT_HEAD(&head, tsqe, entries);
            }
        }
    }

#ifdef DEBUG
    printf("State_transition:\n");

    printf("state\\character: ");
    for (int i = 0; i < SIGMA_SIZE; i++) {
        printf("%2c ", 'a' + i);
    }
    printf("\n");

    for (int i = 0; i < MAX_NUM_OF_STATES; i++) {
        printf("%15d: ", i);
        for (int j=0; j< SIGMA_SIZE; j++) {
            int transition = State_transition[i][j];
            (transition != -1)? printf("%2d ", transition): printf(" - ");
        }
        printf("\n");
    }
    printf("\n");

    printf("State_supply:\n");
    printf("state:  ");
    for (int i = 0; i < MAX_NUM_OF_STATES; i++) {
        printf("%2d ", i);
    }
    printf("\n");
    printf("supply: ");
    for (int i = 0; i < MAX_NUM_OF_STATES; i++) {
        int supply = State_supply[i];
        (supply != -1)? printf("%2d ", State_supply[i]): printf(" - ");
    }
    printf("\n\n");

    printf("State_output:\n");
    printf("state:  ");
    for (int i = 0; i < MAX_NUM_OF_STATES; i++) {
        printf("%2d ", i);
    }
    printf("\n");
    printf("output: ");
    for (int i = 0; i < MAX_NUM_OF_STATES; i++) {
        int output = State_output[i];
        (output != 0)? printf("%2d ", output): printf(" - ");
    }
    printf("\n\n");
#endif

    return states;
}


// find next state using transition and supply functions
__device__ trie_state findNextState(trie_state const * const d_state_transition, const size_t d_state_transition_pitch , trie_state const * const d_state_supply, const trie_state currentState, const char nextInput)
{
    trie_state result = currentState;
    const int ch = nextInput - 'a';

    // If transition is not defined, use supply function 
    trie_state* element = (trie_state*)((char*)d_state_transition + result * d_state_transition_pitch) + ch;
    while (*element == UINT32_MAX) {
        result = d_state_supply[result];
        element = (trie_state*)((char*)d_state_transition + result * d_state_transition_pitch) + ch;
    }
    return *element;
}

trie_state findNextStateHost(trie_state const * const d_state_transition, const size_t d_state_transition_pitch, trie_state const * const d_state_supply, const trie_state currentState, const char nextInput)
{
    trie_state result = currentState;
    const int ch = nextInput - 'a';

    // If transition is not defined, use supply function 
    trie_state* element = (trie_state*)((char*)d_state_transition + result * d_state_transition_pitch) + ch;
    while (*element == UINT32_MAX) {
        result = d_state_supply[result];
        element = (trie_state*)((char*)d_state_transition + result * d_state_transition_pitch) + ch;
    }
    return *element;
}

unsigned int computeOnDevice(char* h_searchphase)
{
    const size_t OUT_SIZE = NUM_BLOCKS * BLOCK_DIM;

    // number of matches per thread
    unsigned int* h_out; 
    char* d_searchphase;
    trie_state* d_state_transition;
    trie_state* d_state_supply;
    trie_state* d_state_output;
    unsigned int* d_out;
    size_t state_transition_pitch;
    cudaEvent_t start, stop;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    h_out = (unsigned int*)malloc(OUT_SIZE * sizeof(unsigned int));

    // using cudamallocPitch to prvent coallsing memory when allocating 2d memory
    gpuErrchk(cudaMallocPitch(&d_state_transition, &state_transition_pitch, SIGMA_SIZE * sizeof(trie_state), MAX_NUM_OF_STATES));
    gpuErrchk(cudaMalloc(&d_state_supply, MAX_NUM_OF_STATES * sizeof(trie_state)));
    gpuErrchk(cudaMalloc(&d_state_output, MAX_NUM_OF_STATES * sizeof(trie_state)));
    gpuErrchk(cudaMalloc(&d_searchphase, N * sizeof(char)));
    gpuErrchk(cudaMalloc(&d_out, OUT_SIZE * sizeof(unsigned int)));


    gpuErrchk(cudaMemcpy2D(d_state_transition, state_transition_pitch, &State_transition, SIGMA_SIZE * sizeof(trie_state), SIGMA_SIZE * sizeof(trie_state), MAX_NUM_OF_STATES, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_state_supply, State_supply, MAX_NUM_OF_STATES * sizeof(trie_state), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_state_output, State_output, MAX_NUM_OF_STATES * sizeof(trie_state), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_searchphase, h_searchphase, N * sizeof(char), cudaMemcpyHostToDevice));

#ifdef DEBUG
    //CPU TEST
    int results = 0;
    trie_state currentState = 0;
    for (size_t i = 0; i < strlen(h_searchphase); i++) {
        currentState = findNextStateHost(*State_transition, SIGMA_SIZE * sizeof(trie_state), State_supply, currentState, h_searchphase[i]);

        if (State_output[currentState] == 0) {
            //printf("(%d) see: %c\t move to: %d\n", i, h_searchphase[i], currentState);
            continue;
        }

        //printf("see: %c\t move to: %d\n", h_searchphase[i], currentState);

        results++;
    }

    printf("CPU TEST result: %d\n", results);
#endif


    gpuErrchk(cudaEventRecord(start));
    AhoCorasickKernel <<<NUM_BLOCKS, BLOCK_DIM / GRANUALITY >>> (d_searchphase, d_state_transition, state_transition_pitch, d_state_supply, d_state_output ,d_out);
    gpuErrchk(cudaEventRecord(stop));

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_out, d_out, OUT_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    unsigned int final_result = 0;
    for (int i = 0; i < OUT_SIZE; i++) {
        final_result += h_out[i];
    }

    float milliseconds = 0;
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Execution time GPU (sec): %f \n", milliseconds / 1000);

    /*gpuErrchk(cudaFree(d_searchphase));
    gpuErrchk(cudaFree(d_state_transition));
    gpuErrchk(cudaFree(d_state_supply));
    gpuErrchk(cudaFree(d_state_output));
    gpuErrchk(cudaFree(d_out));*/

    gpuErrchk(cudaDeviceReset());

    return (final_result);
}

__global__ void AhoCorasickKernel(char const * const d_searchphase, trie_state const * const d_state_transition, const size_t state_transition_pitch,
    trie_state const * const d_state_supply, trie_state const* const d_state_output, unsigned int * const d_out)
{
    size_t blockId = blockIdx.x;
    size_t threadId = threadIdx.x;

#ifdef CAST_TO_INT
    __shared__ uchar4 s_searchphase[S_MEMSIZE];
#else
    __shared__ unsigned char s_searchphase[S_MEMSIZE];
#endif
    unsigned int results = 0;
    
    // define start and stop auxilary variables.
    size_t d_start, d_stop; //used to indicate the input string positions where the search phase begins and ends respectively
    size_t s_start, s_stop;
    d_start = (blockId * BLOCK_START_STEP + threadId * THREAD_START_STEP);
    d_stop = d_start + THREAD_WORK_LENGTH;
    s_start = threadId * THREAD_START_STEP;
    s_stop = s_start + THREAD_WORK_LENGTH;

#ifdef CAST_TO_INT
    // prevent meory coalesceing and bandwith problems by reading N/ S_MEMSIZE chunks by each
    // thread in the block into shared memory.
    // increase global memory bandwidth utilization by reading from searchphase as a unit4 -
    // reading a total of 16 bytes per word, and thus using a segment size of 128
    // access to each 
    uint4* uint4_text = reinterpret_cast<uint4*>(d_searchphase);

//#pragma unroll 16
    for (size_t i = d_start; i < d_stop; i+=16)
    {
        uint4 uint4_var = uint4_text[i / 16];
        s_searchphase[threadIdx.x * 4 + 0] = *reinterpret_cast<uchar4*>(&uint4_var.x);
        s_searchphase[threadIdx.x * 4 + 1] = *reinterpret_cast<uchar4*>(&uint4_var.y);
        s_searchphase[threadIdx.x * 4 + 2] = *reinterpret_cast<uchar4*>(&uint4_var.z);
        s_searchphase[threadIdx.x * 4 + 3] = *reinterpret_cast<uchar4*>(&uint4_var.w);
        /*s_searchphase[s_start + j + 0] = uchar4_var0.x;
        s_searchphase[s_start + j + 1] = uchar4_var0.y;
        s_searchphase[s_start + j + 2] = uchar4_var0.z;
        s_searchphase[s_start + j + 3] = uchar4_var0.w;
        s_searchphase[s_start + j + 4] = uchar4_var1.x;
        s_searchphase[s_start + j + 5] = uchar4_var1.y;
        s_searchphase[s_start + j + 6] = uchar4_var1.z;
        s_searchphase[s_start + j + 7] = uchar4_var1.w;
        s_searchphase[s_start + j + 8] = uchar4_var2.x;
        s_searchphase[s_start + j + 9] = uchar4_var2.y;
        s_searchphase[s_start + j + 10] = uchar4_var2.z;
        s_searchphase[s_start + j + 11] = uchar4_var2.w;
        s_searchphase[s_start + j + 12] = uchar4_var3.x;
        s_searchphase[s_start + j + 13] = uchar4_var3.y;
        s_searchphase[s_start + j + 14] = uchar4_var3.z;
        s_searchphase[s_start + j + 15] = uchar4_var3.w;*/
    }

    //add last thread overlap
    /*for (int i = 0; i < overlap && d_stop + i < phase_size-4; i+=4)
    {
        uint4 uint4_var = uint4_text[d_stop + i / 16];
        s_searchphase[s_stop/16 + i + 0] = *reinterpret_cast<uchar4*>(&uint4_var.x);
        s_searchphase[s_stop/16 + i + 1] = *reinterpret_cast<uchar4*>(&uint4_var.y);
        s_searchphase[s_stop/16 + i + 2] = *reinterpret_cast<uchar4*>(&uint4_var.z);
        s_searchphase[s_stop/16 + i + 3] = *reinterpret_cast<uchar4*>(&uint4_var.w);
    }*/
#else
    for (size_t i = d_start, j = 0; i < d_stop && i < N; ++i, ++j)
    {
        s_searchphase[s_start + j] = d_searchphase[i];
    }

    //add last thread overlap
    for (int i = 0; i < OVERLAP && d_stop + i < N; ++i)
    {
        s_searchphase[s_stop + i] = d_searchphase[d_stop + i];
    }
#endif
    __syncthreads();


    // use overlap to find patterns between 2 thread's blocks.
    // add overlap, only if you are not the last thread in the last block.
    // otherwise memory will move out of bound.
    // use multiplication by bool expresion to avoid thread divergence.
    s_stop += OVERLAP * (d_stop + OVERLAP < N);
    
    trie_state currentState = 0;


#ifdef CAST_TO_INT
#pragma unroll 4
    for (size_t i = s_start; i < s_stop; i+=4) {
        currentState = findNextState(d_state_transition, state_transition_pitch, d_state_supply, currentState, s_searchphase[i].x);

        if (d_state_output[currentState] == 0) {
            continue;
        }

        results++;

        /*currentState = findNextState(d_state_transition, state_transition_pitch, d_state_supply, currentState, s_searchphase[i].y);

        if (d_state_output[currentState] == 0) {
            continue;
        }

        results++;

        currentState = findNextState(d_state_transition, state_transition_pitch, d_state_supply, currentState, s_searchphase[i].z);

        if (d_state_output[currentState] == 0) {
            continue;
        }

        results++;

        currentState = findNextState(d_state_transition, state_transition_pitch, d_state_supply, currentState, s_searchphase[i].w);

        if (d_state_output[currentState] == 0) {
            continue;
        }

        results++;*/
    }

    d_out[blockId * blockDim.x + threadId] = results;
#else
#pragma unroll 4
    for (size_t i = s_start; i < s_stop && i < N; i++) {
        currentState = findNextState(d_state_transition, state_transition_pitch, d_state_supply ,currentState, s_searchphase[i]);

        if (d_state_output[currentState] == 0) {
            continue;
        }
        
        results++;
    }

    d_out[blockId * blockDim.x + threadId] = results;
#endif
}

unsigned int computeGold(const char* target, const char* P[NUM_OF_PATTERNS])
{
    clock_t start, end;
    unsigned int result;

    struct ahocorasick aho;
    int id[NUM_OF_PATTERNS] = { 0 };
    //char* target = "Lorem ipsum dolor sit amet, consectetur brown elit. Proin vehicula brown egestas. Aliquam a dui tincidunt, elementum sapien in, ultricies lacus. Phasellus congue, sapien nec";
    aho_init(&aho);

#ifdef SIMPLE_EXAMPLE
    id[0] = aho_add_match_text(&aho, P[0], strlen(P[0]));
    id[1] = aho_add_match_text(&aho, P[1], strlen(P[1]));
    id[2] = aho_add_match_text(&aho, P[2], strlen(P[2]));
    id[3] = aho_add_match_text(&aho, P[3], strlen(P[3]));
#else
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
#endif

    aho_create_trie(&aho);
#ifdef DEBUG
    aho_register_match_callback(&aho, callback_match_pos, (void*)target);
#endif

    // use aho-corasick algorithm from https://github.com/morenice/ahocorasick
    start = clock();
    result = aho_findtext(&aho, target, strlen(target));
    end = clock();

    printf("total match CPU:%u\n", result);

    double milliseconds = ((double)(end - start)) / double(CLOCKS_PER_SEC);
    printf("Execution Time Gold CPU (sec): %lf\n", milliseconds);

    return result;
}

void compareWithGold(const unsigned int result, const unsigned int reference)
{
    unsigned int result_regtest = result - reference;
    printf("\n\nTest %s\n", (0 == result_regtest) ? "PASSED" : "FAILED");
    printf("device: %d  host: %d\n", result, reference);
}


