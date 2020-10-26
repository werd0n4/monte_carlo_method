#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "sequence/monte_carlo.hpp"


int main()
{
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}