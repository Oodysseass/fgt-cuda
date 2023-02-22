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

    return 0;
}
