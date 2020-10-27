
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <ctime>

//do random GPU
#include <curand.h>
#include <curand_kernel.h>

//Zahardcodowane dane: przedzial A-B 1-4, funkcja 2*x-4
float user_f_host(float x) {
	return 2 * x - 4;
}

__device__ double user_f(double x) {
	return 2 * x - 4;
}

//czesc odpowiedzialna za handling errorów w czeœci GPU ********************
#define gpuErrchk(ans){gpuAssert((ans), __FILE__,__LINE__);}
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
//**************************************************************


__global__ void randomPointCuda(double* result) {
	/* CUDA's random number library uses curandState_t to keep track of the seed value
	 we will store a random state for every thread  */
	curandState_t state;

	/* we have to initialize the state */
	curand_init(0, /* the seed controls the sequence of random values that are produced */
		0, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

	/* curand works like rand - except that it takes a state as a parameter */
	//*result = curand(&state) % MAX;
}

__global__ void parrelCuda(int gpu_size,unsigned long seed, double A, double B, double min_Y, double max_Y, int* c, int size) {
	//random w oparciu o dane
	//ifContains ma sprawdzic czy nalezy
	//wynik idzie do tablicy c (-1,0,1)

	//1.losuj random x z przedzialu A do B
	//2.losuj random y z przedzialu min_y max_Y
	//3. policz funkcje f(x)---------------------poziom if contains
	//4. przyrównaj f(x) z wylosowanym Y
	//5. zwróc wynik
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < size) {
		int addToScore = 0;
		curandState_t state;
		curand_init(seed, gid, 0, &state);

		for (int j = 0; j < gpu_size; j++) {


			//1.
			//int X = ((double)(curand(&state) % (int)(B - A))) + A;
			double X = ((double)(curand(&state) / (double)(0x0FFFFFFFFUL) * (B - A))) + A;
			//2.
			//int Y = ((double)(curand(&state) % (int)(max_Y - min_Y))) + min_Y;
			double Y = ((double)(curand(&state) / (double)(0x0FFFFFFFFUL) * (max_Y - min_Y))) + min_Y;
			//printf("GPU X: %f  Y: %f \n",X,Y);///tutaj tylko 1 i 3


			if ((Y > 0) && (Y <= user_f(X))) {
				addToScore += 1;
			}
			else if ((Y < 0) && (Y >= user_f(X))) {
				addToScore += -1;
			}
			//else ->addToScore= 0;


		}
		c[gid] = addToScore;
	}
}


__global__ void hello()
{	
	printf("I am GPU: Hello World !\n");
	
}

__global__ void sum_array_gpu(int* a, int* b, int* c, int size) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < size) {
		c[gid] = a[gid] + b[gid];
	}
}

int main() {
	float user_A, user_B, randomY_min, randomY_max, result;
	int n, score;
	int gpu_size = 100000; //100;// 100000;//1;//100000;
	//result to pole ca³ki
	//obecnie zahardcodowany przedzia³ ca³kowania
	user_A = 1;
	user_B = 4;
	unsigned long cuRand_seed = time(NULL);
	//Dok³adnoœæ- liczba n losowanych punktów-tutaj potem ustawiæ znacznie wiêcej
	//Obecnie dotyczy jedynie losowania punktów w polu, a nie granic pola
	//n = 200;



	//TODO piotrek part
	//LOSOWANIE PUNKTOW w celu wyznaczenia tylko min i max Y "prostokata"
	//opis dzia³ania: wylosuj N punktow, dziêki temu wyznacz granice Y prostok¹ta- zrównolegliæ w tym miejscu
	randomY_min = -2;
	randomY_max = 4;// Niech losuj (X), jesli f(X)>Max ustaw nowy max, jesli f(x)<Min ustaw Min




	cudaError error;
	int size = 10000;//2* 100000;//10000;//to jest nowe n
	int block_size = 1024;//1024;//128;

	int NO_BYTES = size * sizeof(int);
	//hostp ointers
	int * gpu_results;
	//allocate memory for host pointers
	gpu_results = (int*)malloc(NO_BYTES);

	memset(gpu_results, 0, NO_BYTES);//wypelnij gpu_results zerami, o rozmiarze nobytes

	//device pointers
	int * d_c;
	gpuErrchk(cudaMalloc((int**)&d_c, NO_BYTES));
	
	//start grid
	dim3 block(block_size);
	dim3 grid((size / block.x) + 1);

	parrelCuda << <grid, block >> > (gpu_size,cuRand_seed,user_A, user_B, randomY_min, randomY_max, d_c, size);
	cudaDeviceSynchronize();

	//memory transfer back to host!!!
	cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);

	
	printf("CPU: obliczenia dotarly az tutaj!\n");
	//printf("sum array cpu execution time: %4.6f \n",
	//	(double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));
	score = 0;
	for (int i = 0; i < size; i++) {
		score += gpu_results[i];
		//printf("Score= %d", score);
	}
	result = (score / ((double)size*gpu_size)) *
		((user_B - user_A) * (randomY_max - randomY_min));
	
	printf("\nWartosc calki wynosi w przyblizeniu %f\n", result);
	printf("Wartosc zsumowania wynosi w przyblizeniu %d\n", score);
	printf("Wartosc stosunku %f\n", (score / ((double)size*gpu_size)));
	cudaFree(d_c);
	free(gpu_results);


	int device_number = 0;
	cudaDeviceProp iProp;
	cudaGetDeviceProperties(&iProp, device_number);
	printf("Device %d: %s\n", device_number, iProp.name);
	printf("Number of multiprocessors:   %d\n", iProp.multiProcessorCount);
	printf("Clockrate: %d\n", iProp.clockRate);
	printf("maxThreadsPerBlock: %d\n", iProp.maxThreadsPerBlock);
	printf("maxThreadsDim max dimension of a block x %d\n", iProp.maxThreadsDim[0]);
	printf("maxThreadsDim max dimension of a block y %d\n", iProp.maxThreadsDim[1]);
	printf("maxThreadsDim max dimension of a block z %d\n", iProp.maxThreadsDim[2]);
	printf("maxGridSize[3] max size of grid dimension x : %d\n", iProp.maxGridSize[0]);
	printf("maxGridSize[3] max size of grid dimension y : %d\n", iProp.maxGridSize[1]);
	printf("maxGridSize[3] max size of grid dimension z : %d\n", iProp.maxGridSize[2]);
	return 0;

}




int ifContains(float x, float y) {//funcIN
	if ((y > 0) && (y <= user_f_host(x)))
		return 1;
	else if ((y > 0) && (y <= user_f_host(x)))
		return -1;
	return 0;
}

//random number from a to b
//TODO
double randomPoint(double a, double b) {
	return a + (double)rand() / (double)(RAND_MAX + 1) * (b - a);
}


int main23() {
	float user_A, user_B, randomY_min, randomY_max, result;
	int n, score;
	//result to pole ca³ki
	//obecnie zahardcodowany przedzia³ ca³kowania
	user_A = 1;
	user_B = 4;

	//Dok³adnoœæ- liczba n losowanych punktów-tutaj potem ustawiæ znacznie wiêcej
	//Obecnie dotyczy jedynie losowania punktów w polu, a nie granic pola
	n = 500;



	//TODO piotrek part
	//LOSOWANIE PUNKTOW w celu wyznaczenia tylko min i max Y "prostokata"
	//opis dzia³ania: wylosuj N punktow, dziêki temu wyznacz granice Y prostok¹ta- zrównolegliæ w tym miejscu
	randomY_min = -2;
	randomY_max = 4;// Niech losuj (X), jesli f(X)>Max ustaw nowy max, jesli f(x)<Min ustaw Min


	//TODO 
	//Tutaj drugie zrównoleglenie- losowaæ na GPU 10000000 punktów i sprawdzaæ wartoœæ ifContains
	//trzeba jakoœ przerobiæ i randompoint- uzyæ cuda random
	//i ifContains- zapisaæ gdzieœ wartoœæi przedzia³ów aby GPU ca³ czas mia³o do nich dostêp?
	score = 0;//co to
	srand((unsigned)time(NULL));
	for (int i = 0; i < n; i++) {
		score += ifContains(randomPoint(user_A, user_B), randomPoint(randomY_min, randomY_max));
	}

	//to czesc ktora juz szeregowo
	result = (score / (double)n) *
		((user_B - user_A) * (randomY_max - randomY_min));

	printf("Wartosc calki wynosi w przyblizeniu %f\n", result);

	return 0;
}