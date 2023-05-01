#include <time.h> // time library

void reset_timer(int clock_starts);           // record the current clock ticks
double elapsed_time_in_sec(int clock_starts); // returns time in seconds from the start
void show_elapsed_time_in_sec(int clock_starts);
// MEMORY
// Given (allowed): malloc_c(size_t) strdup_c(const char*)
// Allowed string functions: strcpy, strncpy, strlen, strcmp, strncmp
// Unallowed memory functions: memcpy, memccpy, memmove, wmemmove,
//    or other direct memory copy/move functions
//    these functions performs 'BLOCKED' operations so that
//    a large chunk of memory allocation or move operation are
//    efficiently implemented, so they break UNIT TIME assumption
//    in algorithm design
// Unallowed string functions: strdup

// to compute used memory
// use malloc_c defined below, instead of malloc, calloc, realloc, etc.
// malloc_c accumulates the size of the dynamically allocated memory to
// global/static variable used_memory, so that we can measure the

size_t used_memory_in_bytes(size_t used_memory); // returns the number of bytes allocated by malloc_c and strdup_c

void *malloc_c(size_t size, size_t *used_memory);
void *calloc_c(int temp, size_t size, size_t *used_memory);
char *strdup_c(const char *s, size_t used_memory); // create a duplicate word with counting bytes
