#include <iostream>
#include "../headers/mtx.hpp"
#include "../headers/fglt.hpp"

std::ostream &output(std::ostream &outfile, int **arr, int rows, int cols)
{

    outfile
        << "\"Node id (0-based)\","
        << "\"[0] vertex (==1)\","
        << "\"[1] degree\","
        << "\"[2] 2-path\","
        << "\"[3] bifork\","
        << "\"[4] 3-cycle\","
        << std::endl;

    for (int i = 0; i < rows; i++)
    {
        outfile << i << ",";
        for (int j = 0; j < cols - 1; j++)
            outfile << arr[j][i] << ",";
        outfile << arr[cols - 1][i] << std::endl;
    }
    return outfile;
}

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

    // output
    std::fstream ofnet("freq_net.csv", std::ios::out);

    if (ofnet.is_open())
    {
        output(ofnet, nfreq, adjacent->rows, 5);
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
