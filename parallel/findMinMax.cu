#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <stdio.h>
#include <utility>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>

typedef double(*FunctionCallback)(double);

namespace parallel {

__global__ void minMaxThread(int n, double *generatedNumbers, double *devMin, double *devMax, FunctionCallback func, double a, double b)
{
	int i = threadIdx.x;
	if (i < n)
    {
		double x = generatedNumbers[i] * (b - a) + a;
		double value = func(x);
		devMax[i] = ( value > devMax[i] ) ? value : devMax[i];
		devMin[i] = ( value < devMin[i] ) ? value : devMin[i];
	}
}

void minMaxKernelsStart(int n, double *generatedNumbers, double *devMin, double *devMax, FunctionCallback func, double a, double b)
{
	int device_number = 0;
	cudaDeviceProp iProp;
	cudaGetDeviceProperties(&iProp, device_number);

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

	curandGenerateUniformDouble(gen, generatedNumbers, n);

	minMaxThread<<<(n + iProp.maxThreadsPerBlock - 1) / iProp.maxThreadsPerBlock, iProp.maxThreadsPerBlock>>>
		(n, generatedNumbers, devMin, devMax, func, a, b);
	cudaDeviceSynchronize();
}

std::pair<double, double> minMaxValue(int n, double a, double b, FunctionCallback func)
{
	const int S = 1<<20; //2^20

	double *generatedNumbers, *devMin, *devMax, *hostMin, *hostMax;
	int loopSize, arraySize;

	loopSize = (( n + S - 1) / S);
	arraySize = S;

	hostMin = (double *)malloc(S * sizeof(double));
	hostMax = (double *)malloc(S * sizeof(double));


	cudaMalloc((void**)&devMin, S * sizeof(double));
	cudaMalloc((void**)&devMax, S * sizeof(double));
	cudaMalloc((void**)&generatedNumbers, S * sizeof(double));

	std::fill(hostMin, hostMin + arraySize, std::numeric_limits<double>::max());
	std::fill(hostMax, hostMax + arraySize, std::numeric_limits<double>::lowest());

	cudaMemcpy(devMin, hostMin, arraySize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devMax, hostMax, arraySize * sizeof(double), cudaMemcpyHostToDevice);

	for (int i = 0; i < loopSize; i++)
	{
		minMaxKernelsStart(arraySize, generatedNumbers, devMin, devMax, func, a, b);
	}

	cudaMemcpy(hostMin, devMin, arraySize * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostMax, devMax, arraySize * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(generatedNumbers);
    cudaFree(devMin);
	cudaFree(devMax);

	auto pair = std::make_pair(	*std::min_element(hostMin, hostMin + arraySize), 
								*std::max_element(hostMax, hostMax + arraySize));

	free(hostMin);
	free(hostMax);

	return pair;
}

__global__ void minMaxThreadV2(unsigned long seed, double a, double b, double *devMin, double *devMax, int size, int calculationsPerThread, FunctionCallback f)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
	{
		curandState_t state;
		curand_init(seed, idx, 0, &state);
		double x, y;
		double min = devMin[idx];
		double max = devMax[idx];
		for(int i =0; i< calculationsPerThread; i++)
		{
			x = curand_uniform_double(&state) * (b - a) + a;
			y = f(x);

			max = ( y > max ) ? y : max;
			min = ( y < min ) ? y : min;

		}
		devMin[idx] = min;
		devMax[idx] = max;

	}
}


std::pair<double, double> minMaxValueV2(int n, double a, double b, FunctionCallback func)
{
	unsigned long cuRand_seed = time(NULL);	

	cudaDeviceProp iProp;
	cudaGetDeviceProperties(&iProp, 0);
	//int threads = 512;
	//int blocks = 100;
	int threads = iProp.maxThreadsPerBlock;
	int blocks = iProp.multiProcessorCount;

	int size = threads * blocks;
	int sizeInBytes = size * sizeof(double);

	double *devMin, *devMax, *hostMin, *hostMax;

	cudaMalloc((void**)&devMin, sizeInBytes * sizeof(double));
	cudaMalloc((void**)&devMax, sizeInBytes * sizeof(double));

	hostMin = (double *)malloc(sizeInBytes);
	hostMax = (double *)malloc(sizeInBytes);

	std::fill(hostMin, hostMin + size, std::numeric_limits<double>::max());
	std::fill(hostMax, hostMax + size, std::numeric_limits<double>::lowest());

	cudaMemcpy(devMin, hostMin, sizeInBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(devMax, hostMax, sizeInBytes, cudaMemcpyHostToDevice);

	int calculationsPerThread = (n + size -1) / size;

	minMaxThreadV2<<<blocks, threads>>>(cuRand_seed, a, b, devMin, devMax, size, calculationsPerThread, func);
	cudaDeviceSynchronize();


	cudaMemcpy(hostMin, devMin, sizeInBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostMax, devMax, sizeInBytes, cudaMemcpyDeviceToHost);

	cudaFree(devMin);
	cudaFree(devMax);

	auto pair = std::make_pair(	*std::min_element(hostMin, hostMin + size), 
								*std::max_element(hostMax, hostMax + size));

	free(hostMin);
	free(hostMax);

	return pair;
}





void timeTestMinMaxPar(int m, int n, double a, double b, FunctionCallback f){
    std::cout << std::setprecision(5);
    std::chrono::duration<double> total = std::chrono::duration<double>::zero();
    std::chrono::duration<double> diff;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

	std::ofstream file;
	std::stringstream filename;
	filename << "minMaxPar_" << m << '_' << n << ".txt";
	n = 1 << n;
	file.open(filename.str());
	if (file.good() == true)
	{

    	std::cout << "Testing parallel MinMax... for size: " << n << std::endl;
    	for(int i = 1; i <= m; ++i){
    	    start = std::chrono::high_resolution_clock::now();
    	    minMaxValue(n, a, b, f);
    	    end = std::chrono::high_resolution_clock::now();
    	    std::cout << "\r" << i * 100.0 / m << "%  ";
    	    std::cout << std::flush;
    	    diff = end - start;
			file << diff.count() << std::endl;
			total += diff;
		}
	file.close();
	}

    std::cout << std::endl;
    std::cout << "Parallel MinMax average time: " << total.count()/m << std::endl;
}

}



