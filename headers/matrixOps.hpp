#ifndef MATRIXOPS_HPP
#define MATRIXOPS_HPP

#include <cuda_runtime_api.h>
#include "mtx.hpp"

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            exit(1);                                                   \
        }                                                              \
    }

/**
 * Calculates d0 frequency
 *
 * @param e pointer to array representing d0
 * @param N number of nodes in graph
 */
__global__ void calcdZero(int *e, int N);

/**
 * Calculates d1 frequency
 *
 * @param A  pointer of adjacent matrix in CSR format
 * @param p1 pointer to array representing d1
 */
__global__ void calcdOne(CSRMatrix *A, int *p1);

/**
 * Calculates d2 frequency
 *
 * @param A  pointer of adjacent matrix in CSR format
 * @param p2 pointer to array representing d2
 */
__global__ void calcdTwo(CSRMatrix *A, int *p1, int *p2);

/**
 * Calculates d3 frequency
 *
 * @param A  pointer of adjacent matrix in CSR format
 * @param p1 pointer to frequency d1 needed for calculation
 * @param d3 pointer to array representing d3
 */
__global__ void calcdThree(CSRMatrix *A, int *p1, int *d3);

/**
 * Calculates c3 = A .* A ^ 2
 *
 * @param A  pointer matrix in CSR format
 * @param c3 pointer to result of multiplication
 */
__global__ void calcCThree(CSRMatrix *A, CSRMatrix *c3);

/**
 * Computes d0-d4 frequencies for a graph
 *
 * @param adjacent CSRMatrix pointer reprsenting adjacent matrix of graph
 * @param freq     2d array representing frequencies for each node
 */
__host__ void compute(CSRMatrix *adjacent, int **freq);

#endif
