#ifndef JACOBIGPU_CUH
#define JACOBIGPU_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include<string>
#define TileSize 32
class JacobiGpu{
private:
	const int maxIterations = 40000;
	double *x, *nextX, *A, *b;
	double *dX, *dNextX, *dA, *dB;
	int size;
	int N;
	int type = 2;
	__global__ friend void jacobiIterationWithSharedMemory(double* nextX, const double* __restrict__ A, const double* __restrict__  x, const double* __restrict__ b, int rowSize, int colSize);
public:
	JacobiGpu(int dimension);
	void freeAllMemory();
	void solve(double eps);
	void input(std::string rFile);
	void output(std::string wfile);
};

#endif