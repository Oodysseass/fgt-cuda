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
    CSRMatrix *adjacent;

    // get adjacent matrix
    CSCMatrix tempAdjacent = readMTX(argv[1]);
    adjacent = new CSRMatrix(tempAdjacent.rows, tempAdjacent.columns,
                                tempAdjacent.nz);
    convert(tempAdjacent, adjacent);

    std::cout << "#Rows/Columns: " << adjacent->rows << std::endl;
    std::cout << "#Non-zeros: " << adjacent->nz << std::endl;

    std::cout << "Allocate freq" << std::endl;
    // allocate memory for frequencies table
    int **freq = new int*[5];
    for (int i = 0; i < 5; i++)
        freq[i] = new int[adjacent->rows];

    compute(adjacent, freq);

    // deallocate host memory
    delete[] adjacent->rowIndex;
    delete[] adjacent->nzIndex;
    delete[] adjacent->nzValues;

    for (int i = 0; i < 5; i++)
        delete[] freq[i];
    delete[] freq;

    return 0;
}
