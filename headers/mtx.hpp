#ifndef MTX_HPP
#define MTX_HPP

#include <fstream>
#include <iostream>
#include <string.h>

/**
 * Struct denoting the adjacent matrix of a graph in CSR format
 */
struct CSRAdjacentMatrix
{
    int size;       //# of rows (= columns in our case)
    int nz;         //# of non-zero elements in sparse matrix
    int *rowIndex;  //starting and ending indexes of each row
    int *nzIndex;   //columns of non-zero elements for corresponding rows

    /**
     * CSRAdjacentMatrix constructor
     * 
     * @param size  # of rows of the matrix 
     * @param nz    # of non-zero elements of the matrix 
     */
    CSRAdjacentMatrix(int size, int nz);
};

/**
 * Struct denoting the adjacent matrix of a graph in CSC format
 */
struct CSCAdjacentMatrix
{
    int size;       //# of columns (= rows in our case)
    int nz;         //# of non-zero elements in sparse matrix
    int *colIndex;  //starting and ending indexes of each column
    int *nzIndex;   //columns of non-zero elements for corresponding rows

    /**
     * CSCAdjacentMatrix constructor
     * 
     * @param size  # of columns of the matrix 
     * @param nz    # of non-zero elements of the matrix 
     */
    CSCAdjacentMatrix(int size, int nz);
};

/**
 * Constructs adjacent matrix in CSR format from .mtx file
 * 
 * @param filename filename of the .mtx file to read
 * @return a struct representing the adjacent matrix in CSR
 */
CSRAdjacentMatrix readMTX(std::string filename);

/**
 * Converts from CSC to CSR
 * 
 * @param A CSC struct to covert to CSR
 * @return the converted matrix in CSR
 */
CSRAdjacentMatrix convert(CSCAdjacentMatrix A);

#endif