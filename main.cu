#include <utility>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>

#include "parallel/findMinMax.cu"
#include "parallel/monte_carlo.cu"
#include "sequence/findMinMax.cpp"
#include "sequence/monte_carlo.cpp"



typedef double(*FunctionCallback)(double);

__host__ void print_device_info();
__device__ double myCosDev(double x);
double myCos(double x);
void timeTestMonteCarloSeq(int m, int n, double a, double b, double min, double max, FunctionCallback f);
void timeTestMonteCarloPar(int m, int n, double a, double b, double min, double max, FunctionCallback f);
void timeTestMinMaxloSeq(int m, int n, double a, double b, FunctionCallback f);
void timeTestMinMaxPar(int m, int n, double a, double b, FunctionCallback f);

__device__  FunctionCallback myCosPar = myCosDev;



int main()
{
	print_device_info();

	FunctionCallback myCosHost;
	cudaMemcpyFromSymbol(&myCosHost, myCosPar, sizeof(FunctionCallback));

	double a =0;
	double b =10;
	double min = -1;
	double max = 1;

    int n = 3;
    int m = 1<<23;

	sequence::timeTestMonteCarloSeq(n, m, a, b, min, max, myCos);
	parallel::timeTestMonteCarloPar(n, m, a, b, min, max, myCosHost);
	sequence::timeTestMinMaxloSeq(n, m, a, b, myCos);
	parallel::timeTestMinMaxPar(n, m, a, b, myCosHost);
	

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
	return cos(x);
}

double myCos(double x)
{
	return cos(x);
}


