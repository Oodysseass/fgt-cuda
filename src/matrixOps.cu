#include "../headers/matrixOps.hpp"

__global__ void calcdZero(int *e, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N)
        e[i] = 1;
}

__global__ void calcdOne(CSRMatrix *A, int *p1)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < A->rows)
        p1[i] = A->rowIndex[i + 1] - A->rowIndex[i];
}

__global__ void calcdTwo(CSRMatrix *A, int *p1, int *p2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < A->rows)
    {
        for (int j = A->rowIndex[i]; j < A->rowIndex[i + 1]; j++)
            p2[i] += p1[A->nzIndex[j]];

        p2[i] -= p1[i];
    }
}