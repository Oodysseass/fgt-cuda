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

__host__ void compute(CSRMatrix *adjacent, int **freq)
{
    // allocate device memory
    CSRMatrix *devAdjacent;
    int **devFreq;

    std::cout << "Allocate devAdjacent" << std::endl;
    CHECK_CUDA(cudaMalloc(&devAdjacent, sizeof(CSRMatrix)))
    CHECK_CUDA(cudaMalloc(&devAdjacent->rowIndex,
                          (adjacent->rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc(&devAdjacent->nzIndex,
                          (adjacent->nz) * sizeof(int)))
    CHECK_CUDA(cudaMalloc(&devAdjacent->nzValues,
                          (adjacent->nz) * sizeof(int)))

    std::cout << "Allocate devFreq" << std::endl;
    CHECK_CUDA(cudaMalloc((void **)&devFreq, 5 * sizeof(int *)))
    for (int i = 0; i < 5; i++)
        CHECK_CUDA(cudaMalloc((void **)&devFreq[i],
                              (adjacent->rows) * sizeof(int)))

    std::cout << "Copy adjacent" << std::endl;
    // copy to device
    CHECK_CUDA(cudaMemcpy(devAdjacent->rowIndex, adjacent->rowIndex,
                          (adjacent->rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(devAdjacent->nzIndex, adjacent->nzIndex,
                          (adjacent->nz) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(devAdjacent->nzValues, adjacent->nzValues,
                          (adjacent->nz) * sizeof(int),
                          cudaMemcpyHostToDevice))

    std::cout << "Free device" << std::endl;
    // deallocate device memory
    CHECK_CUDA(cudaFree(devAdjacent->rowIndex))
    CHECK_CUDA(cudaFree(devAdjacent->nzIndex))
    CHECK_CUDA(cudaFree(devAdjacent->nzValues))

    for (int i = 0; i < 5; i++)
        CHECK_CUDA(cudaFree(devFreq[i]))
    CHECK_CUDA(cudaFree(devFreq))
    std::cout << "end" << std::endl;
}
