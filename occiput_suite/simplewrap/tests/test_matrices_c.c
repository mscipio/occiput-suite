
// SimpleWrap - Simpler wrapper of C libraries based on Ctypes 
// Stefano Pedemonte
// Aalto University, School of Science, Helsinki
// Oct 2013, Helsinki 


#include <stdio.h>
#include <stdlib.h>

#define SUCCESS     0
#define INPUT_ERROR 1


typedef struct matrix{
    int ndim; 
    int *dim; 
    void *data; 
} matrix; 


/*
Sum two matrices - single precision
*/
extern int sum_matrices_f(matrix *A, matrix *B, matrix *out) 
{
    int i,j; 
    int ndim = A->ndim;
    if (A->ndim != B->ndim || out->ndim)
        return INPUT_ERROR; 
    
    for (i=0; i<ndim; i++)
        if (A->dim[i] != B->dim[i] || B->dim[i] != out->dim[i])
            return INPUT_ERROR; 
    for (i=0; i<ndim ;i++)
        for (j=0; j<A->dim[i] ;j++)
            ((float*) out->data)[i] = ((float*) A->data)[i] + ((float*) B->data)[i]; 
    return SUCCESS; 
}

extern int swrap_sum_matrices_f(char *description)
{
    const char d[] = " [{'name':'A',  'type':'matrix','dtype':'float','direction':'in' }, \
                        {'name':'B',  'type':'matrix','dtype':'float','direction':'in' }, \
                        {'name':'out','type':'matrix','dtype':'float','direction':'out'} ] "; 
    sprintf(description, d);
    return SUCCESS; 
}


/*
Sum two matrices - double precision
*/
extern int sum_matrices_d(matrix *A, matrix *B, matrix *out) 
{
    int i,j; 
    int ndim = A->ndim;
    if (A->ndim != B->ndim || out->ndim)
        return INPUT_ERROR; 
    
    for (i=0; i<ndim; i++)
        if (A->dim[i] != B->dim[i] || B->dim[i] != out->dim[i])
            return INPUT_ERROR; 
    for (i=0; i<ndim ;i++)
        for (j=0; j<A->dim[i] ;j++)
            ((double*) out->data)[i] = ((double*) A->data)[i] + ((double*) B->data)[i]; 
    return SUCCESS; 
}

extern int swrap_sum_matrices_d(char *description)
{
    const char d[] = " [{'name':'A',  'type':'matrix','dtype':'double','direction':'in' }, \
                        {'name':'B',  'type':'matrix','dtype':'double','direction':'in' }, \
                        {'name':'out','type':'matrix','dtype':'double','direction':'out'} ] "; 
    sprintf(description, d);
    return SUCCESS; 
}

extern int swrap_list_functions(char *functions)
{
    const char f[] = " ['sum_matrices_f','sum_matrices_d'] ";
    sprintf(functions, f);
    return SUCCESS;
}



