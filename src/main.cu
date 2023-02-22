#include <iostream>
#include "../headers/mtx.hpp"
#include "../headers/matrixOps.hpp"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: ./main <filename>" << std::endl;
        return 1;
    }

    // ~~~~~~~ variable declaration and memory allocation
    CSRMatrix *adjacent, *p1;

    // get adjacent matrix
    CSCMatrix tempAdjacent = readMTX(argv[1]);
    adjacent = new CSRMatrix(tempAdjacent.rows, tempAdjacent.columns,
                                tempAdjacent.nz);
    convert(tempAdjacent, adjacent);

    std::cout << "#Rows/Columns: " << adjacent->rows << std::endl;
    std::cout << "#Non-zeros: " << adjacent->nz << std::endl;

    // allocate memory for frequencies table
    int **freq = new int*[5];
    for (int i = 0; i < 5; i++)
        freq[i] = new int[adjacent->rows];

    // allocate device memory
    CSRMatrix *devAdjacent;
    int **devFreq;

    CHECK_CUDA(cudaMalloc((void **)&devAdjacent->rowIndex,
                          (adjacent->rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devAdjacent->nzIndex,
                          (adjacent->nz) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&devAdjacent->nzValues,
                          (adjacent->nz) * sizeof(int)))

    CHECK_CUDA(cudaMalloc((void **)&devFreq, 5 * sizeof(int*)))
    for (int i = 0; i < 5; i++)
        CHECK_CUDA(cudaMalloc((void **)&devFreq[i],
                              (adjacent->rows) * sizeof(int)))

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

    // deallocate host memory
    delete[] adjacent->rowIndex;
    delete[] adjacent->nzIndex;
    delete[] adjacent->nzValues;

    for (int i = 0; i < 5; i++)
        delete[] freq[i];
    delete[] freq;

    // deallocate device memory
    CHECK_CUDA(cudaFree(devAdjacent->rowIndex))
    CHECK_CUDA(cudaFree(devAdjacent->nzIndex))
    CHECK_CUDA(cudaFree(devAdjacent->nzValues))

    for (int i = 0; i < 5; i++)
        CHECK_CUDA(cudaFree(devFreq[i]))
    CHECK_CUDA(cudaFree(devFreq))

    return 0;
}
