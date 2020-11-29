#include <utility>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <ostream>
#include <sstream>
#include "parallel/findMinMax.cu"
#include "parallel/monte_carlo.cu"
#include "sequence/findMinMax.cpp"
#include "sequence/monte_carlo.cpp"



typedef double(*FunctionCallback)(double);

__host__ void print_device_info();
__device__ double myCosDev(double x);
double myCos(double x);


__device__  FunctionCallback myCosPar = myCosDev;



int main()
{
	print_device_info();

	FunctionCallback myCosHost;
	cudaMemcpyFromSymbol(&myCosHost, myCosPar, sizeof(FunctionCallback));

	double a =0;
	double b =10;
	auto minMax = sequence::minMaxValue(1 << 20, a, b, myCos);
	double min = minMax.first;
	double max = minMax.second;
	

    int n = 50;

	for (int m = 15; m <= 25; m++)
	{
		std::cout << "-------------------------------------------------------------------------------" << std::endl;
		//sequence::timeTestMonteCarloSeq(n, m, a, b, min, max, myCos);
		//parallel::timeTestMonteCarloPar(n, m, a, b, min, max, myCosHost);
		std::cout << "-------------------------------------------------------------------------------" << std::endl;
		//sequence::timeTestMinMaxloSeq(n, m, a, b, myCos);
		parallel::timeTestMinMaxPar(n, m, a, b, myCosHost);
	}
	for (int m = 26; m <= 30; m++)
	{
		parallel::timeTestMinMaxPar(n, m, a, b, myCosHost);
		//parallel::timeTestMonteCarloPar(n, m, a, b, min, max, myCosHost);
	}

    return 0;
}


__host__ void print_device_info() {
	int device_number = 0;
	cudaDeviceProp iProp;

	cudaGetDeviceProperties(&iProp, device_number);

	printf("Device %d: %s\n", device_number, iProp.name);
	printf("Compute capability: %d.%d\n", iProp.major, iProp.minor);
	printf("Number of multiprocessors:   %d\n", iProp.multiProcessorCount);
	printf("Clockrate: %d\n", iProp.clockRate);
	printf("maxThreadsPerBlock: %d\n", iProp.maxThreadsPerBlock);

	printf("Total amount of global memory: %4.2f KB\n", iProp.totalGlobalMem / 1024.0);
	printf("Total amount of constant memory: %4.2f KB\n", iProp.totalConstMem / 1024.0);
	printf("Total amount of shared memory per block: %4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
	printf("Total amount of shared memory per MP: %4.2f KB\n", iProp.sharedMemPerMultiprocessor / 1024.0);

	printf("maxThreadsDim max dimension of a block x %d\n", iProp.maxThreadsDim[0]);
	printf("maxThreadsDim max dimension of a block y %d\n", iProp.maxThreadsDim[1]);
	printf("maxThreadsDim max dimension of a block z %d\n", iProp.maxThreadsDim[2]);

	printf("maxGridSize[3] max size of grid dimension x : %d\n", iProp.maxGridSize[0]);
	printf("maxGridSize[3] max size of grid dimension y : %d\n", iProp.maxGridSize[1]);
	printf("maxGridSize[3] max size of grid dimension z : %d\n", iProp.maxGridSize[2]);
}

__device__ double myCosDev(double x)
{
	//return (exp(x) * sin(x) + pow(x, 5.0)) / log(x);
	return cos(x);
}

double myCos(double x)
{
	//return (exp(x) * sin(x) + pow(x, 5.0)) / log(x);
	return cos(x);
}


