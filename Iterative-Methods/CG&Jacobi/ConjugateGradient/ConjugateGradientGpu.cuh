#ifndef CONJUGATEGRADIENTGPU_CUH
#define CONJUGATEGRADIENTGPU_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include<string>
// vecVec
#define BLOCK_DIM_VEC 32
//matVec
#define NB_ELEM_MAT 32
#define BLOCK_SIZE_MAT 32

#define b(x) (b[(x)])
class ConjugateGradientGpu {
private:
	const int maxIterations = 1000;
	float * hA, *hb, *hx, *h_r_norm;
	float* dA, *db, *dx, *dr, *dAp, *dp;
	float* dBeta, *dAlpha, *d_r_norm, *d_r_norm_old, *d_temp_scal;
	int size;
	__global__ friend void matVec2(float* A, float* b, float* out);
	__global__ friend void vecPlusVec2(float* a, float* b, float* out);
	__global__ friend void vecMinVec(float* a, float* b, float* out);
	__global__ friend void scalarVec(float* scalar, float* a, float* out);
	__global__ friend void vecVec2(float* a, float* b, float* out);
	__global__ friend void memCopy(float* in, float* out);
	__global__ friend void divide(float* num, float* den, float* out);
public:
	ConjugateGradientGpu(int dimension);
	void freeAllMemory();
	void solve(double eps);
	void solveCG_cuda(double eps);
	void input(std::string rFile);
	void output(std::string wfile);
};

#endif