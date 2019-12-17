#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <time.h>
#include <fstream>
#include<string>
#include"Util.h"
#include"ConjugateGradient.cuh"
//extern "C" {
//#include "helper.h"
//#include "sequential.h"
//}
using namespace std;

/*
 * --More efficient implementation--
 * Computes a (square) symmetric matrix vector product
 * Input: pointer to 1D-array-stored matrix, 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
__global__ void matVec2(float* A, float* b, float* out,int size) {
	__shared__ float b_shared[NB_ELEM_MAT];

	int effective_block_width;
	if ((blockIdx.x + 1) * NB_ELEM_MAT <= size) {
		effective_block_width = NB_ELEM_MAT;
	}
	else {
		// needed to avoid overflow in next row
		effective_block_width = size % NB_ELEM_MAT;
	}

	if (threadIdx.x < effective_block_width)
		b_shared[threadIdx.x] = b[blockIdx.x * NB_ELEM_MAT + threadIdx.x];

	__syncthreads();

	int idy = blockIdx.y *  BLOCK_SIZE_MAT + threadIdx.x;
	float tmp_scal = 0.0;
	// threads outside matrix dimension are not needed (vertical)
	if (idy < size) {
		for (int i = 0; i < effective_block_width; i++) {
			// take advantage of symmetric matrix for coalesced memory access
			tmp_scal += b_shared[i] * A[(blockIdx.x * NB_ELEM_MAT + i)*size + (idy)];
		}
		atomicAdd(out + idy, tmp_scal);
	}
}

/*
 * Computes the sum of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the sum in memory at the location of the pointer out
 */

__global__ void vecPlusVec(float* a, float* b, float* out,int size) {
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < size) {
		out[index_x] = b[index_x] + a[index_x];
	}
}

/*
 * Computes the sum of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the sum in memory at the location of the pointer out
 * Also 0's the vector b
 */
__global__ void vecPlusVec2(float* a, float* b, float* out, int size) {
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < size) {
		out[index_x] = b[index_x] + a[index_x];
		b[index_x] = 0.0;
	}
}

/*
 * Computes the difference of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the sum in memory at the location of the pointer out
 */
__global__ void vecMinVec(float* a, float* b, float* out, int size) {
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < size) {
		out[index_x] = a[index_x] - b[index_x];
	}
}

/*
 * --More efficient implementation--
 * Computes the inner product of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
__global__ void vecVec2(float* a, float* b, float* out, int size) {
	// each block has it's own shared_tmp of size BLOCK_DIM_VEC
	__shared__ float shared_tmp[BLOCK_DIM_VEC];

	// needed for atomicAdd
	if (threadIdx.x + blockDim.x * blockIdx.x == 0) {
		*out = 0.0;
	}


	if (blockIdx.x * blockDim.x + threadIdx.x < size) {
		shared_tmp[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x]
			* b[blockIdx.x * blockDim.x + threadIdx.x];
	}
	else {
		// needed for the reduction
		shared_tmp[threadIdx.x] = 0.0;
	}

	// reduction within block
	for (int i = blockDim.x / 2; i >= 1; i = i / 2) {
		// threads access memory position written by other threads so sync is needed
		__syncthreads();
		if (threadIdx.x < i) {
			shared_tmp[threadIdx.x] += shared_tmp[threadIdx.x + i];
		}
	}

	// atomic add the partial reduction in out
	if (threadIdx.x == 0) {
		atomicAdd(out, shared_tmp[0]);
	}
}

/*
 * Computes the product of a scalar with a vector
 * Input: pointer to scalar, pointer to 1D-array-stored vector
 * Stores the sum in memory at the location of the pointer out
 */
__global__ void scalarVec(float* scalar, float* a, float* out, int size) {
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < size) {
		out[index_x] = a[index_x] * *scalar;
	}
}

/*
 * Copies the content of vector in to vector out
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 */
__global__ void memCopy(float* in, float* out, int size) {
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < size) {
		out[index_x] = in[index_x];
	}
}

/*
 * Computes the quotient of 2 scalars
 * Input: pointer to scalar, pointer to scalar
 * Stores the quotient in memory at the location of the pointer out
 */
__global__ void divide(float* num, float* den, float* out, int size) {
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x == 0) {
		*out = *num / *den;
	}
}

/*
 * Main CG solver
 * All the given pointers are device pointers, with correct initial values
 */
void ConjugateGradientGpu::solveCG_cuda(double eps) {

	dim3 vec_block_dim(BLOCK_DIM_VEC);
	dim3 vec_grid_dim((size + BLOCK_DIM_VEC - 1) / BLOCK_DIM_VEC);

	dim3 mat_grid_dim((size + NB_ELEM_MAT - 1) / NB_ELEM_MAT, (size +  BLOCK_SIZE_MAT - 1) /  BLOCK_SIZE_MAT);
	dim3 mat_block_dim( BLOCK_SIZE_MAT);

	vecVec2 << <vec_grid_dim, vec_block_dim >> > (dr, dr, d_r_norm_old, size);

	int multicity = int(0.1 / eps);
	CpuTimer timer = CpuTimer();
	timer.start();
	int k = 0;
	while ((k < maxIterations) && (*h_r_norm > eps)) {
		// temp = A * p (only compute matrix vector product once)
		matVec2 << <mat_grid_dim, mat_block_dim >> > (dA, dp, dAp, size);

		// alpha_k = ...
		vecVec2 << <vec_grid_dim, vec_block_dim >> > (dp, dAp, d_temp_scal,size);
		divide << <1, 1 >> > (d_r_norm_old, d_temp_scal, dAlpha, size);

		// r_{k+1} = ...
		scalarVec << <vec_grid_dim, vec_block_dim >> > (dAlpha, dAp, dAp, size);
		vecMinVec << <vec_grid_dim, vec_block_dim >> > (dr, dAp, dr, size);

		// x_{k+1} = ...
		scalarVec << <vec_grid_dim, vec_block_dim >> > (dAlpha, dp, dAp, size);
		vecPlusVec << <vec_grid_dim, vec_block_dim >> > (dx, dAp, dx, size);

		// beta_k = ...
		vecVec2 << <vec_grid_dim, vec_block_dim >> > (dr, dr, d_r_norm, size);
		divide << <1, 1 >> > (d_r_norm, d_r_norm_old, dBeta, size);

		// p_{k+1} = ...
		scalarVec << <vec_grid_dim, vec_block_dim >> > (dBeta, dp, dAp, size);
		vecPlusVec2 << <vec_grid_dim, vec_block_dim >> > (dr, dAp, dp, size);

		// set r_norm_old to r_norm
		memCopy << <1, 1 >> > (d_r_norm, d_r_norm_old, size);

		// copy to r_norm to CPU (to evaluate stop condition)
		cudaMemcpy(h_r_norm, d_r_norm, sizeof(float), cudaMemcpyDeviceToHost);

		if (*h_r_norm < eps*multicity) {
			std::cout << timer.stop() << " ";
			multicity = int(multicity / 10);
		}
		k++;
	}
	std::cout << std::endl << "Iterations:" << k << std::endl;
}

void ConjugateGradientGpu::freeAllMemory() {
	// cleanup memory host
	free(hA);
	free(hb);
	free(hx);
	free(h_r_norm);

	cudaFree(dAlpha);
	cudaFree(dBeta);
	cudaFree(d_r_norm);
	cudaFree(d_r_norm_old);
	cudaFree(d_temp_scal);

	// cleanup memory device
	cudaFree(dA);
	cudaFree(db);
	cudaFree(dx);
	cudaFree(dp);
	cudaFree(dr);
	cudaFree(dAp);
}
ConjugateGradientGpu::ConjugateGradientGpu(int dimension) {

	this->size = dimension;
	// allocate host memory
	hA = (float *)malloc(sizeof(float) * size * size);
	hb = (float *)malloc(sizeof(float) * size);
	hx = (float *)malloc(size * sizeof(float));
	h_r_norm = (float *)malloc(sizeof(float));
	*h_r_norm = 1.0;

	// allocate device memory
	cudaMalloc((void **)&dA, size * size * sizeof(float));
	cudaMalloc((void **)&db, size * sizeof(float));
	cudaMalloc((void **)&dx, size * sizeof(float));
	cudaMalloc((void **)&dp, size * sizeof(float));
	cudaMalloc((void **)&dr, size * sizeof(float));
	cudaMalloc((void **)&dAp, size * sizeof(float));

	cudaMalloc((void **)&dBeta, sizeof(float));
	cudaMalloc((void **)&dAlpha, sizeof(float));
	cudaMalloc((void **)&d_r_norm, sizeof(float));
	cudaMalloc((void **)&d_r_norm_old, sizeof(float));
	cudaMalloc((void **)&d_temp_scal, sizeof(float));
}
void ConjugateGradientGpu::input(std::string rFile) {
	int n = size;
	ifstream  fin(rFile);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			fin >> hA[i*n + j];
			//cout << A[i][j] << " ";
		}
		//cout << endl;
	}
	for (int i = 0; i < n; i++) {
		fin >> hb[i];
		hx[i] = 0;
	}
	fin.close();
}
void ConjugateGradientGpu::output(std::string wfile) {
	int n = size;
	ofstream fout(wfile);
	for (int i = 0; i < n; i++)
	{
		fout << hx[i] << "   ";
		//cout << x[i] << "   ";
	}
	fout.close();
}

void ConjugateGradientGpu::solve(double eps) {
	// copy host memory to device
	cudaMemcpy(dA, hA, size * size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dx, hx, size * sizeof(float), cudaMemcpyHostToDevice);
	// assume x0 = 0
	cudaMemcpy(dp, hb, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dr, hb, size * sizeof(float), cudaMemcpyHostToDevice);

	//float elapsedTime = 0.0;
	//cudaEvent_t event_start, event_stop;
	//cudaEventCreate(&event_start);
	//cudaEventCreate(&event_stop);
	//cudaEventRecord(event_start, 0);

	solveCG_cuda(eps);

	//cudaEventRecord(event_stop, 0);
	//cudaEventSynchronize(event_stop);
	//cudaEventElapsedTime(&elapsedTime, event_start, event_stop);
	// allocate memory for the result on host side
	cudaDeviceSynchronize();
	// copy result from device to host
	cudaMemcpy(hx, dx, sizeof(float) * size, cudaMemcpyDeviceToHost);

	//assert(moreOrLessEqual(hx, h_x_seq) == 1);

}
void main(int argc, char ** argv) {
	int dimension = stoi(argv[1], 0, 10);
	ConjugateGradientGpu * cGG = new ConjugateGradientGpu(dimension);
	cGG->input(argv[2]);
	double eps = stod(argv[3]);
	cGG->solve(eps);
	cGG->output(argv[4]);
	cGG->freeAllMemory();
}