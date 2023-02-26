#ifndef FGLT_HPP
#define FGLT_HPP

#include <cuda_runtime_api.h>
#include <sys/time.h>
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
 * Transforms raw frequencies to net
 *
 * @param f0  pointer to f0 raw frequency
 * @param f1  pointer to f1 raw frequency
 * @param f2  pointer to f2 raw frequency
 * @param f3  pointer to f3 raw frequency
 * @param f4  pointer to f4 raw frequency
 * @param nf0 pointer to f0 net frequency
 * @param nf1 pointer to f1 net frequency
 * @param nf2 pointer to f2 net frequency
 * @param nf3 pointer to f3 net frequency
 * @param nf4 pointer to f4 net frequency
 * @param N   number of vertices
 */
__global__ void rawToNet(int *f0, int *f1, int *f2, int *f3, int *f4,
                         int *nf0, int *nf1, int *nf2, int *nf3, int *nf4,
                         int N);

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
