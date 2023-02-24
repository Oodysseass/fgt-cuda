#include "../headers/mtx.hpp"

CSRMatrix::CSRMatrix(int rows, int columns, int nz)
{
    this->rows = rows;
    this->columns = columns;
    this->nz = nz;
    rowIndex = new int[rows + 1]();
    nzIndex = new int[nz]();
    nzValues = new int[nz]();
}

CSCMatrix::CSCMatrix(int columns, int rows, int nz)
{
    this->columns = columns;
    this->rows = rows;
    this->nz = nz;
    colIndex = new int[columns + 1]();
    nzIndex = new int[nz]();
    nzValues = new int[nz]();
}

CSCMatrix readMTX(std::string filename)
{
    // open the file
    std::ifstream fin(filename);
    if (fin.fail())
    {
        std::cerr << "File " << filename << " could not be opened! Aborting..." << std::endl;
        exit(1);
    }

    char mmx[20], b1[20], b2[20], b3[20], b4[20];
    int numRows, numCols, numNZ, size;
    bool isSymmetric;

    // read banner
    fin >> mmx >> b1 >> b2 >> b3 >> b4;

    // parse banner
    if (strcmp(b1, "matrix"))
    {
        std::cerr << "Currently works only with 'matrix' option, aborting..." << std::endl;
        exit(1);
    }
    if (strcmp(b2, "coordinate"))
    {
        std::cerr << "Currently works only with 'coordinate' option, aborting..." << std::endl;
        exit(1);
    }
    if (strcmp(b3, "pattern"))
    {
        std::cerr << "Currently works only with 'pattern' format, aborting..." << std::endl;
        exit(1);
    }

    if (!strcmp(b4, "symmetric"))
        isSymmetric = true;

    // skip comment
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    while (fin.peek() == '%')
        fin.ignore(2048, '\n');

    // read defining parameters
    fin >> numRows >> numCols >> numNZ;

    if (isSymmetric)
        size = 2 * numNZ;
    else
        size = numNZ;

    // allocate space for COO format
    int *rowcoo = new int[size];
    int *colcoo = new int[size];

    // read the COO data
    int j = 0;
    for (int i = 0; i < numNZ; i++)
    {
        fin >> rowcoo[j] >> colcoo[j];

        if (isSymmetric)
            // we do not keep self-loop, remove edges
            if (rowcoo[j] == colcoo[j])
                size -= 2;

            // put symmetric edge
            else
            {
                rowcoo[j + 1] = colcoo[j];
                colcoo[j + 1] = rowcoo[j];
                j += 2;
            }
        // we do not keep self-loop, remove edge
        else if (rowcoo[j] == colcoo[j])
            size -= 1;
        else
            j++;
    }

    numNZ = size;
    rowcoo = static_cast<int *>(realloc(rowcoo, numNZ * sizeof(int)));
    colcoo = static_cast<int *>(realloc(colcoo, numNZ * sizeof(int)));

    // close connection to file
    fin.close();

    // convert coo to csc
    CSCMatrix cscAdj = CSCMatrix(numCols, numRows, numNZ);

    // find column sizes
    for (int i = 0; i < numNZ; i++)
    {
        cscAdj.colIndex[colcoo[i] - 1]++;
        cscAdj.nzValues[i] = 1;
    }

    for (int i = 0, cumsum = 0; i < numRows; i++)
    {
        int temp = cscAdj.colIndex[i];
        cscAdj.colIndex[i] = cumsum;
        cumsum += temp;
    }
    cscAdj.colIndex[numCols] = numNZ;

    // copy row indices to correct place
    for (int i = 0; i < numNZ; i++)
    {
        int coli = colcoo[i] - 1;
        int dest = cscAdj.colIndex[coli];
        cscAdj.nzIndex[dest] = rowcoo[i] - 1;

        cscAdj.colIndex[coli]++;
    }

    // revert column pointers
    for (int i = 0, last = 0; i < numCols; i++)
    {
        int temp = cscAdj.colIndex[i];
        cscAdj.colIndex[i] = last;

        last = temp;
    }

    // deallocate
    free(rowcoo);
    free(colcoo);
    return cscAdj;
}

void convert(CSCMatrix A, CSRMatrix *B)
{
    int dest, temp, last = 0, cumsum = 0;

    for (int i = 0; i < A.rows + 1; i++)
        B->rowIndex[i] = 0;

    for (int i = 0; i < A.nz; i++)
    {
        B->rowIndex[A.nzIndex[i]]++;
        B->nzValues[i] = 1;
    }

    for (int i = 0; i < A.rows; i++)
    {
        temp = B->rowIndex[i];
        B->rowIndex[i] = cumsum;
        cumsum += temp;
    }
    B->rowIndex[A.rows] = A.nz;

    for (int i = 0; i < A.rows; i++)
    {
        for (int j = A.colIndex[i]; j < A.colIndex[i + 1]; j++)
        {
            temp = A.nzIndex[j];
            dest = B->rowIndex[temp];

            B->nzIndex[dest] = i;
            B->rowIndex[temp]++;
        }
    }

    for (int i = 0; i < A.rows + 1; i++)
    {
        temp = B->rowIndex[i];
        B->rowIndex[i] = last;
        last = temp;
    }
}