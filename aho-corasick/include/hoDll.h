// hoDll.h - Contains declarations of math functions
#pragma once

#ifdef hoDll_EXPORTS
#define AHO_API __declspec(dllexport)
#else
#define AHO_API __declspec(dllimport)
#endif

#include <stdbool.h>


// Initialize a Fibonacci relation sequence
// such that F(0) = a, F(1) = b.
// This function must be called before any other function.

#define MAX_AHO_CHILD_NODE 256 /* Character 1 byte => 256 */

extern "C" AHO_API struct aho_text_t
{
    int id;
    char* text;
    int len;
    struct aho_text_t* prev, * next;
};


extern "C" AHO_API struct aho_trie_node
{
    unsigned char text;
    unsigned int ref_count;

    struct aho_trie_node* parent;
    struct aho_trie_node* child_list[MAX_AHO_CHILD_NODE];
    unsigned int child_count;

    bool text_end;
    struct aho_text_t* output_text; /* when text_end is true */

    struct aho_trie_node* failure_link;
    struct aho_trie_node* output_link;
};


extern "C" AHO_API struct aho_trie
{
    struct aho_trie_node root;
};

extern "C" AHO_API void fibonacci_init(
    const unsigned long long a, const unsigned long long b);

extern "C" AHO_API struct aho_match_t
{
    int id;
    unsigned long long pos;
    int len;
};

extern "C" AHO_API struct ahocorasick
{
#define AHO_MAX_TEXT_ID INT_MAX
    int accumulate_text_id;
    struct aho_text_t* text_list_head;
    struct aho_text_t* text_list_tail;
    int text_list_len;

    struct aho_trie trie;

    void (*callback_match)(void* arg, struct aho_match_t*);
    void* callback_arg;
};

extern "C" AHO_API struct aho_queue_node
{
    struct aho_queue_node* next, * prev;
    struct aho_trie_node* data;
};

extern "C" AHO_API struct aho_queue
{
    struct aho_queue_node* front;
    struct aho_queue_node* rear;
    unsigned int count;
};


extern "C" AHO_API void aho_init(struct ahocorasick* aho);
extern "C" AHO_API void aho_destroy(struct ahocorasick* aho);

extern "C" AHO_API int aho_add_match_text(struct ahocorasick* aho, const char* text, unsigned int len);
extern "C" AHO_API bool aho_del_match_text(struct ahocorasick* aho, const int id);
extern "C" AHO_API void aho_clear_match_text(struct ahocorasick* aho);

extern "C" AHO_API void aho_create_trie(struct ahocorasick* aho);
extern "C" AHO_API void aho_clear_trie(struct ahocorasick* aho);

extern "C" AHO_API unsigned int aho_findtext(struct ahocorasick* aho, const char* data, unsigned long long data_len);

extern "C" AHO_API void aho_register_match_callback(struct ahocorasick* aho,
    void (*callback_match)(void* arg, struct aho_match_t*),
    void* arg);

/* for debug */
extern "C" AHO_API void aho_print_match_text(struct ahocorasick* aho);


extern "C" AHO_API void aho_queue_init(struct aho_queue* que);
extern "C" AHO_API void aho_queue_destroy(struct aho_queue* que);

/* inline */
bool aho_queue_empty(struct aho_queue* que);

extern "C" AHO_API bool aho_queue_enqueue(struct aho_queue* que, struct aho_trie_node* node);
extern "C" AHO_API struct aho_queue_node* aho_queue_dequeue(struct aho_queue* que);

extern "C" AHO_API void aho_init_trie(struct aho_trie* t);
extern "C" AHO_API extern "C" AHO_API void aho_destroy_trie(struct aho_trie* t);

extern "C" AHO_API bool aho_add_trie_node(struct aho_trie* t, struct aho_text_t* text);
extern "C" AHO_API void aho_connect_link(struct aho_trie* t);
extern "C" AHO_API void aho_clean_trie_node(struct aho_trie* t);

extern "C" AHO_API struct aho_text_t* aho_find_trie_node(struct aho_trie_node** start, const unsigned char text);

extern "C" AHO_API void aho_print_trie(struct aho_trie* t);

extern "C" AHO_API int isItWorking(int x);

/* TODO:
 * bool aho_del_trie_node(struct aho_trie* t, struct aho_text_t* text);
 */

