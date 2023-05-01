/*
  mem_check.c
  memory and execution time check Utilties in C
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>    // string library
#include "mem_check.h" // memory check headerfile

/*
EXECUTION TIME CHECK
*/
// time in seconds from when reset_timer() was called
// TIME
// THE FOLLOWING FUNCTIONS SHOW HOW TO MEASURE THE EXECUTION TIME
// USING A BUILT-IN FUNCTION clock() DEFINED IN time.h
// NOTE: STATIC VARIABLES ARE NECESSARY TO RECORD CLOCKS
// USAGE:
//    reset_timer();	// reset the start time
//    ....		// statements to measure time
//    t = elapsed_time_in_sec();

// global static variable for start clock
void reset_timer(int clocks_starts)
{
  int *ptr;
  ptr = &clocks_starts;
  *ptr = clock(); // record the current clock ticks
}
double elapsed_time_in_sec(int clocks_starts)
// returns time in seconds from the start
{
  return ((double)(clock() - clocks_starts)) / CLOCKS_PER_SEC;
}
void show_elapsed_time_in_sec(int clocks_starts)
{
  double t;
  t = ((double)(clock() - clocks_starts)) / CLOCKS_PER_SEC;
  printf("\n\nTime used: %f sec\n\n", t);
}

/*
 MEMORY CHECK FUNCTION
*/
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

// used amount of memory exactly.
size_t used_memory_in_bytes(size_t used_memory)
// returns the number of bytes allocated by malloc_c and strdup_c
{
  return used_memory;
}

void *malloc_c(size_t size, size_t *used_memory)
{
  if (size > 0)
  {
    // increase the required memory count
    *used_memory += size;
    return malloc(size);
  }
  else
    return NULL;
}
void *calloc_c(int temp, size_t size, size_t *used_memory)
{
  if (size > 0)
  {
    // increase the required memory count
    // printf("size of memory : %ld\n",size);
    for (int i = 0; i < temp; i++)
      *used_memory += size;
    return calloc(temp, size);
  }
  else
    return NULL;
}

// create a duplicate word with counting bytes
char *strdup_c(const char *s, size_t used_memory)
{
  int size;
  size = strlen(s) + 1; // including last null character
  used_memory += size;
  return strdup(s);
}

// DO NOT USE malloc() and strdup()
// the below two lines detects unallowed usage of malloc and strdup
// NULL pointer will be returned, causing runtime errors
// NOTE: '#define' is effective only after declaration
#define malloc NOT_ALLOWED
#define strdup NOT_ALLOWED
#define calloc CHANGING_TO_CALLOC_C
