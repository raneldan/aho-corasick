
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "simpleQ.h"
#include "include/hoDll.h"

//definitions
#define SIMPLE_EXAMPLE
#define DEBUG

#define NUM_BLOCKS 1
#define BLOCK_DIM 1024
#define N 1000000 // size of search string. devisable by NUM_BLOCKS and by BLOCK_DIM
#define S_CHUNK (N / NUM_BLOCKS) // amount of characters in a chunk
#define S_THREAD (N / (NUM_BLOCKS * BLOCK_DIM) + M - 1) // amount of characters proccessed by a thread
#define S_MEMSIZE *

#define trie_state uint32_t

#define SIGMA_SIZE ('z' - 'a' + 1) // small case english letters.

//define list of patterns
#ifdef SIMPLE_EXAMPLE
#define NUM_OF_PATTERNS 4 // count of patterns
const char* P[NUM_OF_PATTERNS] = {
    "he",
    "she",
    "his",
    "hers"
};
// calculate maximum number of states, should be the sum of lengths of patterns.
const size_t MAX_NUM_OF_STATES = 12;
#else
#define NUM_OF_PATTERNS 10 // count of patterns
const char* P[NUM_OF_PATTERNS] = {
    "consectetur",
    "est",
    "egestasaliquam",
    "elementum",
    "ultricies",
    "vehicula",
    "tortor",
    "inauditum",
    "inquam" ,
    "aut"
};
// calculate maximum number of states, should be the sum of lengths of patterns.
const size_t MAX_NUM_OF_STATES = 79;
#endif //SIMPLE_EXAMPLE

// queue defenitions for calculation of supply function
typedef	struct _trie_state_queue_entry {
    SIMPLEQ_ENTRY(_trie_state_queue_entry) entries;
    trie_state state;
};
SIMPLEQ_HEAD(queue_head_type, _trie_state_queue_entry) head;

//function definitions
int preprocessing();
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


void callback_match_pos(void* arg, struct aho_match_t* m)
{
    char* text = (char*)arg;

    printf("match text in CPU:");
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
    char searchphase[N];
    char pattern[NUM_OF_PATTERNS];
    
#ifdef SIMPLE_EXAMPLE
    strcpy_s(searchphase, "ahishers");
#else
    // input
    FILE* fp;

    fp = fopen("Input.txt", "r");
    fgets(searchphase, N, fp);
    fclose(fp);
#endif //SIMPLE_EXAMPLE

#ifdef DEBUG
    printf("searchphase: %s\n\n", searchphase);
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
    //allocateData();

    //computeOnDevice();
    
    //calculateResult();

    computeGold(searchphase, P);

    //compareWithGold();

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
int findNextState(trie_state currentState, char nextInput)
{
    trie_state result = currentState;
    int ch = nextInput - 'a';

    // If transition is not defined, use supply function 
    while (State_transition[result][ch] == UINT32_MAX) {
        result = State_supply[result];
    }
    return State_transition[result][ch];
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
    unsigned int overlap = NUM_OF_PATTERNS - 1; // To ensure the correctness of the results, m−1 overlapping characters are used per thread
    start = (blockId * N) / NUM_BLOCKS + (N * threadId) / (NUM_BLOCKS * blockDim.x);
    stop = start + N / (NUM_BLOCKS * blockDim.x) + overlap;

}

void calculateResult()
{
    int i;
}

void computeGold(const char* target, const char* P[NUM_OF_PATTERNS])
{
    clock_t start, end;

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
    printf("total match CPU:%u\n", aho_findtext(&aho, target, strlen(target)));
    end = clock();

    double milliseconds = ((double)(end - start)) / double(CLOCKS_PER_SEC);
    printf("Execution Time Gold CPU (sec): %lf\n", milliseconds);
}

void compareWithGold()
{
    int i;
}


