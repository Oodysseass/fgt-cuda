#include <iostream>
#include "../headers/mtx.hpp"
#include "../headers/matrixOps.hpp"

__host__ void makeUnit(CSCMatrix *A);

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: ./main <filename>" << std::endl;
        return 1;
    }

    // ~~~~~~~ variable declaration and memory allocation
    CSRMatrix *adjacent, *p1;
    CSCMatrix *unitVector;

    // get adjacent matrix and copy to GPU
    CSCMatrix tempAdjacent = readMTX(argv[1]);
    adjacent = new CSRMatrix(tempAdjacent.rows, tempAdjacent.columns,
                                tempAdjacent.nz);
    convert(tempAdjacent, adjacent);

    std::cout << "#Rows/Columns: " << adjacent->rows << std::endl;
    std::cout << "#Non-zeros: " << adjacent->nz << std::endl;

    return 0;
}

__host__ void makeUnit(CSCMatrix *A)
{
    A->colIndex[0] = 0;
    A->colIndex[1] = A->nz;

    for (int i = 0; i < A->nz; i++)
    {
        A->nzIndex[i] = i;
        A->nzValues[i] = 1;
    }
}