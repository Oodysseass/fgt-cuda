#include "../headers/fglt.hpp"


__global__ void rawToNet(int *f0, int *f1, int *f2, int *f3, int *f4,
                         int *nf0, int *nf1, int *nf2, int *nf3, int *nf4,
                         int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride)
    {
        if (i < N)
        {
            nf0[i] = f0[i];
            nf1[i] = f1[i];
            nf2[i] = f2[i] - 2 * f4[i];
            nf3[i] = f4[i] - f4[i];
            nf4[i] = f4[i];
        }
    }
}

__global__ void dZeroOne(int *rows, int *e, int *p1, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N)
    {
        e[i] = 1;
        p1[i] = rows[i + 1] - rows[i];
    }
}

__global__ void dTwoThree(int *rows, int *cols, int *p1, int *p2, int *d3, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N)
    {
        d3[i] = p1[i] * (p1[i] - 1) / 2;
        for (int j = rows[i]; j < rows[i + 1]; j++)
            p2[i] += p1[cols[j]];

        p2[i] -= p1[i];
    }
}

__global__ void dFour(int *rows, int *cols, int *d4, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N)
    {
        // for each non-zero element in A(i, cols[j])
        // calculate the corresponding element in A^2
        for (int j = rows[i]; j < rows[i + 1]; j++)
        {
            int col = cols[j];
            
            // take advantage of symmetry to use 2 rows
            // instead of row-column
            // to immitate mutiplication
            for (int k = rows[col]; k < rows[col + 1]; k++)
            {
                for (int l = rows[i]; l < rows[i + 1]; l++)
                {
                    // the two rows do not share an element 
                    // in this column for sure
                    if(cols[k] < cols[l]) break;

                    // all elements are equal to 1
                    // every time there is match in corresponding columns
                    // is a succesful addition to the multiplication
                    if (cols[k] == cols[l])
                    {
                        d4[i]++;
                        break;
                    }
                }
            }
        }
        d4[i] /= 2;
    }
}

__host__ void compute(CSRMatrix *adjacent, int **freq)
{
    // declare variable
    int *devRowIndex, *devNzIndex;
    int *devf0, *devf1, *devf2, *devf3, *devf4;
    int *devNf0, *devNf1, *devNf2, *devNf3, *devNf4;
    int threadsPerBlock, blocksPerGrid;

    // allocate adjacent to device
    CHECK_CUDA(cudaMalloc((void **)&devRowIndex,
                          (adjacent->rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devNzIndex,
                          (adjacent->nz) * sizeof(int)))

    // copy to device
    CHECK_CUDA(cudaMemcpy(devRowIndex, adjacent->rowIndex,
                          (adjacent->rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(devNzIndex, adjacent->nzIndex,
                          (adjacent->nz) * sizeof(int),
                          cudaMemcpyHostToDevice))

    // allocate raw frequencies to device
    CHECK_CUDA(cudaMalloc((void **)&devf0, (adjacent->rows) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devf1, (adjacent->rows) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devf2, (adjacent->rows) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devf3, (adjacent->rows) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devf4, (adjacent->rows) * sizeof(int)))

    // allocate net frequencies to device
    CHECK_CUDA(cudaMalloc((void **)&devNf0, (adjacent->rows) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devNf1, (adjacent->rows) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devNf2, (adjacent->rows) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devNf3, (adjacent->rows) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devNf4, (adjacent->rows) * sizeof(int)))

    // prepare for device functions
    threadsPerBlock = 512;
    blocksPerGrid = (adjacent->rows + threadsPerBlock - 1) / threadsPerBlock;

    // d0, d1
    std::cout << "Calculate d0, d1" << std::endl;
    dZeroOne<<<blocksPerGrid, threadsPerBlock>>>(devRowIndex, devf0, devf1, adjacent->rows);

    // d1, d2
    std::cout << "Calculate d2, d3" << std::endl;
    dTwoThree<<<blocksPerGrid, threadsPerBlock>>>(devRowIndex, devNzIndex, devf1, devf2, devf3, adjacent->rows);

    // d4
    std::cout << "Calculate d4" << std::endl;
    dFour<<<blocksPerGrid, threadsPerBlock>>>(devRowIndex, devNzIndex, devf4, adjacent->rows);

    // transform to net
    std::cout << "Calculate net frequencies" << std::endl;
    rawToNet<<<blocksPerGrid, threadsPerBlock>>>(devf0, devf1, devf2, devf3, devf4, devNf0, devNf1, devNf2,
                                                 devNf3, devNf4, adjacent->rows);

    // copy results to host
    CHECK_CUDA(cudaMemcpy(freq[0], devNf0, adjacent->rows * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(freq[1], devNf1, adjacent->rows * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(freq[2], devNf2, adjacent->rows * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(freq[3], devNf3, adjacent->rows * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(freq[4], devNf4, adjacent->rows * sizeof(int), cudaMemcpyDeviceToHost));

    // deallocate device memory
    CHECK_CUDA(cudaFree(devRowIndex))
    CHECK_CUDA(cudaFree(devNzIndex))
    CHECK_CUDA(cudaFree(devf0))
    CHECK_CUDA(cudaFree(devf1))
    CHECK_CUDA(cudaFree(devf2))
    CHECK_CUDA(cudaFree(devf3))
    CHECK_CUDA(cudaFree(devf4))
    CHECK_CUDA(cudaFree(devNf0))
    CHECK_CUDA(cudaFree(devNf1))
    CHECK_CUDA(cudaFree(devNf2))
    CHECK_CUDA(cudaFree(devNf3))
    CHECK_CUDA(cudaFree(devNf4))
}
