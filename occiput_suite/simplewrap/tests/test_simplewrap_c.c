
// SimpleWrap - Simpler wrapper of C libraries based on Ctypes 
// Stefano Pedemonte
// Aalto University, School of Science, Helsinki
// Oct 2013, Helsinki 


#include <stdio.h>
#include <stdlib.h>


/*
out = in
*/
extern int echo(int *in, int *out)
{
    *out = *in; 
    return 0;
}


/*
Simple callback test: call the callback function given as first argument, passing ot it the integer value given as second argument 
*/
typedef int (*ptrFunc)(int); 

extern int callback_test(ptrFunc callback, int *out)
{
    int return_value = callback(*out); 
    return return_value;
}


