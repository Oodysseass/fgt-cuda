#ifndef FGLT_HPP
#define FGLT_HPP

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
 * Calculates d0 and d1 frequency
 *
 * @param rows rows pointer of adjacent matrix in CSR format
 * @param e    pointer to array representing d0
 * @param p1   pointer to array representing d1
 * @param N    number of vertices
 */
__global__ void dZeroOne(int *rows, int *e, int *p1, int N);

/**
 * Calculates d2 frequency
 *
 * @param rows rows pointer of adjacent matrix in CSR format
 * @param cols columns pointer of adjacent matrix in CSR format
 * @param p1   pointer to array representing p1 needed for calculation
 * @param p2   pointer to array representing d2
 * @param d3   pointer to array representing d3
 * @param N    number of vertices
 */
__global__ void dTwoThree(int *rows, int *cols, int *p1, int *p2, int *d3, int N);

/**
 * Calculates c3 = A .* A ^ 2
 *
 * @param rows rows pointer of adjacent matrix in CSR format
 * @param cols columns pointer of adjacent matrix in CSR format
 * @param d4   pointer to array representing d4
 * @param N    number of vertices
 */
__global__ void dFour(int *rows, int *cols, int *d4, int N);

/**
 * Computes d0-d4 frequencies for a graph
 *
 * @param adjacent CSRMatrix pointer reprsenting adjacent matrix of graph
 * @param freq     2d array representing frequencies for each node
 */
__host__ void compute(CSRMatrix *adjacent, int **freq);

#endif
