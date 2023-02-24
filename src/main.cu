#include <iostream>
#include <sys/time.h>
#include "../headers/mtx.hpp"
#include "../headers/matrixOps.hpp"

struct timeval tic(){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv;
}

static double toc(struct timeval begin){
  struct timeval end;
  gettimeofday(&end, NULL);
  double stime = ((double) (end.tv_sec - begin.tv_sec) * 1000 ) +
    ((double) (end.tv_usec - begin.tv_usec) / 1000 );
  stime = stime / 1000;
  return(stime);
}

void rawToNet(int **nf, int **f, int N)
{
    for (int i = 0; i < N; i++)
    {
        nf[0][i] = f[0][i];
        nf[1][i] = f[1][i];
        nf[2][i] = f[2][i] - 2 * f[4][i];
        nf[3][i] = f[3][i] - f[4][i];
        nf[4][i] = f[4][i];
    }
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

    std::cout << "#Rows/Columns: " << adjacent->rows << std::endl;
    std::cout << "#Non-zeros: " << adjacent->nz << std::endl;

    // allocate memory for frequencies table
    int **freq = new int*[5];
    for (int i = 0; i < 5; i++)
        freq[i] = new int[adjacent->rows];

    struct timeval start = tic();

    // calculate
    compute(adjacent, freq);

    // final net results
    int **nfreq = new int*[5];
    for (int i = 0; i < 5; i++)
        nfreq[i] = new int[adjacent->rows];

    rawToNet(nfreq, freq, adjacent->rows);

    printf("Total elapsed time: %.4f sec\n", toc(start));

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
        delete[] freq[i];
    delete[] freq;

    for (int i = 0; i < 5; i++)
        delete[] nfreq[i];
    delete[] nfreq;

    return 0;
}
