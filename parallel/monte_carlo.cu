#pragma once

#include <iostream>
#include <numeric>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <chrono>
#include <iomanip>


typedef double(*FunctionCallback)(double);

namespace parallel {

__global__ void monteCarloThread(unsigned long seed,
                                    double A, double B,
                                    double min_Y, double max_Y,
                                    int* array, int threads_amount, int gpu_size,
									FunctionCallback f)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
	if (gid < threads_amount) {
        int addToScore = 0;
        double X, randomValue, realValue;
		curandState_t state;
		curand_init(seed, gid, 0, &state);

		for(int j = 0; j < gpu_size; ++j){
            X = curand_uniform_double(&state) * (B - A) + A;
			randomValue = curand_uniform_double(&state) * (max_Y - min_Y) + min_Y;

			realValue = f(X);

			if ((randomValue > 0) && (randomValue <= realValue)) {
				++addToScore;
			}
			else if ((randomValue < 0) && (randomValue >= realValue)) {
				--addToScore;
			}
		}
		array[gid] = addToScore;
	}
}

double monteCarlo(int n, double A, double B, double min, double max, FunctionCallback f){

	unsigned long cuRand_seed = time(NULL);	
    int score = 0;
    double result;

	
	
	cudaDeviceProp iProp;
	cudaGetDeviceProperties(&iProp, 0);
	int threads = iProp.maxThreadsPerBlock;
	int blocks = iProp.multiProcessorCount;
    //hostp pointers
	int* gpu_results;
	//device pointers
    int* d_c;
	
	int size = threads * blocks;
	int sizeInBytes = size * sizeof(int);
    gpu_results = (int*)malloc(sizeInBytes);
    

	memset(gpu_results, 0, sizeInBytes);

	cudaMalloc((int**)&d_c, sizeInBytes);

	int calculationsPerThread = (n + size -1) / size;
	

	monteCarloThread<<<blocks, threads>>> (cuRand_seed, A, B, min, max, d_c, size, calculationsPerThread, f);
	cudaDeviceSynchronize();
	cudaMemcpy(gpu_results, d_c, sizeInBytes, cudaMemcpyDeviceToHost);

	score = std::accumulate(gpu_results, gpu_results+size, 0);

	result = (score / ((double)size*calculationsPerThread)) *
		((B - A) * (max - min));
    
	cudaFree(d_c);
	free(gpu_results);

	return result;
}

void timeTestMonteCarloPar(int m, int n, double a, double b, double min, double max, FunctionCallback f){
    std::cout << std::setprecision(5);
    std::chrono::duration<double> total = std::chrono::duration<double>::zero();
    std::chrono::duration<double> diff;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    std::cout << "Testing parallel Monte Carlo..." << std::endl;
    for(int i = 1; i <= m; ++i){
        start = std::chrono::high_resolution_clock::now();
        monteCarlo(n, a, b, min, max, f);
		end = std::chrono::high_resolution_clock::now();
        std::cout << "\r" << i * 100.0 / m << "%  ";
        std::cout << std::flush;
        diff = end - start;
        total += diff;
    }

    std::cout << std::endl;
    std::cout << "Parallel Monte Carlo average time: " << total.count()/m << std::endl;
}

}