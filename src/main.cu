#include <iostream>
#include <cuda_runtime.h>
#include "../headers/mtx.hpp"


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: ./main <filename>" << std::endl;
        return 1;
    }    

    CSRAdjacentMatrix adjacent = readMTX(argv[2]);

    std::cout << "#Rows/Columns: " << adjacent.size << std::endl;
    std::cout << "#Non-zeros: " << adjacent.nz << std::endl;

    return 0;
}
