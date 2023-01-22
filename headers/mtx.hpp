#ifndef MTX_HPP
#define MTX_HPP

#include <fstream>
#include <iostream>
#include <string.h>
#include <limits>

/**
 * Struct denoting a sparse matrix in CSR format
 */
struct CSRMatrix
{
    int rows;       // # of rows
    int columns;    // # of columns
    int nz;         // # of non-zero elements in sparse matrix
    int *rowIndex;  // starting and ending indexes of each row
    int *nzIndex;   // columns of non-zero elements for corresponding rows
    int *nzValues;  // values of non-zero elements

    /**
     * CSRMatrix constructor
     * 
     * @param rows      # of rows of the matrix
     * @param columns   # of columns of the matrix
     * @param nz        # of non-zero elements of the matrix
     */
    CSRMatrix(int rows, int columns, int nz);
};

/**
 * Struct denoting a sparse matrix in CSC format
 */
struct CSCMatrix
{
    int columns;    // # of columns
    int rows;       // # of rows
    int nz;         // # of non-zero elements in sparse matrix
    int *colIndex;  // starting and ending indexes of each column
    int *nzIndex;   // columns of non-zero elements for corresponding rows
    int *nzValues;  // values of non-zero elements

    /**
     * CSCMatrix constructor
     * 
     * @param columns   # of columns of the matrix
     * @param rows      # of rows of the matrix
     * @param nz        # of non-zero elements of the matrix
     */
    CSCMatrix(int columns, int rows, int nz);
};

/**
 * Constructs adjacent matrix in CSR format from .mtx file
 * 
 * @param csrAdj pointer to CSRMatrix struct to save the matrix
 * @param filename filename of the .mtx file to read
 */
CSCMatrix readMTX(std::string filename);

/**
 * Converts from CSC to CSR
 * 
 * @param A CSC struct to covert to CSR
 * @param B CSR struct pointer, where conversion will be saved
 */
void convert(CSCMatrix A, CSRMatrix *B);

#endif