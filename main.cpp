#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "sequence/monte_carlo.cpp"


int main()
{
    cudaDeviceSynchronize();
    return 0;
}