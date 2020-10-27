
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <curand.h>
#include <curand_kernel.h>


#define gpuErrchk(ans){gpuAssert((ans), __FILE__,__LINE__);}
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// template<typename return_type, typename arg_type>
__global__ void monte_carlo_parallel(unsigned long seed,
                                    double A, double B,
                                    double min_Y, double max_Y,
                                    int* array, int threads_amount, int gpu_size
);

__host__ void print_device_info();