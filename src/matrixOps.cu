#include "../headers/matrixOps.hpp"

__global__ void calcdZero(int *e, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N)
        e[i] = 1;
}

__global__ void calcdOne(int *rows, int *p1, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N)
        p1[i] = rows[i + 1] - rows[i];
}

__global__ void calcdTwo(int *rows, int *cols, int *p1, int *p2, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N)
    {
        for (int j = rows[i]; j < rows[i + 1]; j++)
            p2[i] += p1[cols[j]];

        p2[i] -= p1[i];
    }
}

__global__ void calcdThree(int *p1, int *d3, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N)
        d3[i] = p1[i] * (p1[i] - 1) / 2;
}

__global__ void calcCThree(CSRMatrix *A, CSRMatrix *c3)
{

}

__host__ void compute(CSRMatrix *adjacent, int **freq)
{
    // declare variable
    int *devRowIndex, *devNzIndex, *devNzValues;
    int *devf0, *devf1, *devf2, *devf3, *devf4;
    int threadsPerBlock, blocksPerGrid;

    // allocate adjacent to device
    CHECK_CUDA(cudaMalloc((void **)&devRowIndex,
                          (adjacent->rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devNzIndex,
                          (adjacent->nz) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devNzValues,
                          (adjacent->nz) * sizeof(int)))

    // copy to device
    CHECK_CUDA(cudaMemcpy(devRowIndex, adjacent->rowIndex,
                          (adjacent->rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(devNzIndex, adjacent->nzIndex,
                          (adjacent->nz) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(devNzValues, adjacent->nzValues,
                          (adjacent->nz) * sizeof(int),
                          cudaMemcpyHostToDevice))

    // allocate frequencies to device
    CHECK_CUDA(cudaMalloc((void **)&devf0, (adjacent->rows) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devf1, (adjacent->rows) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devf2, (adjacent->rows) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devf3, (adjacent->rows) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devf4, (adjacent->rows) * sizeof(int)))

    // prepare for device functions
    threadsPerBlock = 512;
    blocksPerGrid = (adjacent->rows + threadsPerBlock - 1) / threadsPerBlock;

    // deallocate device memory
    CHECK_CUDA(cudaFree(devRowIndex))
    CHECK_CUDA(cudaFree(devNzIndex))
    CHECK_CUDA(cudaFree(devNzValues))
    CHECK_CUDA(cudaFree(devf0))
    CHECK_CUDA(cudaFree(devf1))
    CHECK_CUDA(cudaFree(devf2))
    CHECK_CUDA(cudaFree(devf3))
    CHECK_CUDA(cudaFree(devf4))
}
