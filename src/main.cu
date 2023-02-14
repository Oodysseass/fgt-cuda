#include <iostream>
#include "../headers/mtx.hpp"
#include "../headers/matrixOps.hpp"

__host__ void makeUnit(CSRMatrix *A);

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

    // make vector e to calculate p1
    CSRMatrix *unitVector = new CSRMatrix(adjacent->columns, 1, adjacent->columns);
    makeUnit(unitVector);
    sparseMult(adjacent, unitVector, p1);

    std::cout << "#Rows/Columns: " << adjacent->rows << std::endl;
    std::cout << "#Non-zeros: " << adjacent->nz << std::endl;

    return 0;
}

__host__ void makeUnit(CSRMatrix *A)
{
    for (int i = 0; i < A->rows; i++)
        A->rowIndex[i + 1] = A->rowIndex[i] + 1;

    for (int i = 0; i < A->nz; i++)
    {
        A->nzIndex[i] = 0;
        A->nzValues[i] = 1;
    }
}