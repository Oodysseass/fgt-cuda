#include <iostream>
#include "../headers/mtx.hpp"
#include "../headers/fglt.hpp"

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

    std::cout << "Running FGLT for " << argv[1] << std::endl;
    std::cout << "#Rows/Columns: " << adjacent->rows << std::endl;
    std::cout << "#Non-zeros: " << adjacent->nz << std::endl;

    // allocate memory for net frequencies
    int **nfreq = new int*[5];
    for (int i = 0; i < 5; i++)
        nfreq[i] = new int[adjacent->rows];


    // calculate
    compute(adjacent, nfreq);

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < adjacent->rows; j++)
            std::cout << nfreq[i][j] << " ";
        std::cout << std::endl;
    }

    // deallocate host memory
    delete[] adjacent->rowIndex;
    delete[] adjacent->nzIndex;
    delete[] adjacent->nzValues;

    for (int i = 0; i < 5; i++)
        delete[] nfreq[i];
    delete[] nfreq;

    return 0;
}
