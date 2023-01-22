#include <iostream>
#include <cuda_runtime.h>
#include "../headers/mtx.hpp"


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: ./main <filename>" << std::endl;
        return 1;
    }

    // variable declaration and memory allocation
    CSRMatrix *adjacent, *p1;
    CSRMatrix *devAdjacent, *devp1;
    CSCMatrix *unitVector;
    CSCMatrix *devUnitVector;
    int blocksPerGrid, threadsPerBlock = 256;

    cudaMallocHost(&adjacent, sizeof(CSRMatrix));
    cudaMallocHost(&p1, sizeof(CSRMatrix));
    cudaMallocHost(&unitVector, sizeof(CSCMatrix));
    cudaMalloc(&devAdjacent, sizeof(CSRMatrix));
    cudaMalloc(&devp1, sizeof(CSRMatrix));
    cudaMalloc(&devUnitVector, sizeof(CSCMatrix));

    // get adjacent matrix and copy to GPU
    readMTX(adjacent, argv[1]);
    cudaMemcpy(devAdjacent, adjacent, sizeof(CSRMatrix), cudaMemcpyHostToDevice);

    std::cout << "#Rows/Columns: " << adjacent->rows << std::endl;
    std::cout << "#Non-zeros: " << adjacent->nz << std::endl;

    return 0;
}
