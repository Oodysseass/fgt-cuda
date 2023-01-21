#include "../headers/mtx.hpp"

CSRAdjacentMatrix::CSRAdjacentMatrix(int size, int nz)
{
    this->size = size;
    this->nz = nz;
    rowIndex = new int[size + 1];
    nzIndex = new int[nz];
}

CSCAdjacentMatrix::CSCAdjacentMatrix(int size, int nz)
{
    this->size = size;
    this->nz = nz;
    colIndex = new int[size + 1];
    nzIndex = new int[nz];
}

CSRAdjacentMatrix readMTX(std::string filename)
{
    // open the file
    std::ifstream fin(filename);
    if (fin.fail())
    {
        std::cerr << "File " << filename << " could not be opened! Aborting..." << std::endl;
        exit(1);
    }

    char mmx[20], b1[20], b2[20], b3[20], b4[20];
    int temp, numCols, numNZ;

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

    while (fin.peek() == '%')
        fin.ignore(2048, '\n');

    // read defining parameters
    fin >> temp >> numCols >> numNZ;

    // convert coo to csc while reading
    CSCAdjacentMatrix cscAdj = CSCAdjacentMatrix(numCols, numNZ);
    for (int i = 0; i < numNZ; i++)
    {
        fin >> cscAdj.nzIndex[i] >> temp;
        cscAdj.nzIndex[i]--;
        cscAdj.colIndex[temp]++;
    }

    for (int i = 0; i < numCols; i++)
        cscAdj.colIndex[i + 1] += cscAdj.colIndex[i];

    // conver to csr
    CSRAdjacentMatrix csrAdj = convert(cscAdj);

    return csrAdj;
}

CSRAdjacentMatrix convert(CSCAdjacentMatrix A)
{
    int dest, temp, last = 0, cumsum = 0;
    CSRAdjacentMatrix B = CSRAdjacentMatrix(A.size, A.nz);

    for (int i = 0; i < A.size + 1; i++)
        B.rowIndex[i] = 0;

    for (int i = 0; i < A.nz; i++)
        B.rowIndex[A.nzIndex[i]]++;

    for (int i = 0; i < A.size; i++)
    {
        temp = B.rowIndex[i];
        B.rowIndex[i] = cumsum;
        cumsum += temp;
    }
    B.rowIndex[A.size] = A.nz;

    for (int i = 0; i < A.size; i++)
    {
        for (int j = A.colIndex[i]; j < A.colIndex[i + 1]; j++)
        {
            temp = A.nzIndex[j];
            dest = B.rowIndex[temp];

            B.nzIndex[dest] = i;
            B.rowIndex[temp]++;
        }
    }

    for (int i = 0; i < A.size + 1; i++)
    {
        temp = B.rowIndex[i];
        B.rowIndex[i] = last;
        last = temp;
    }

    return B;
}