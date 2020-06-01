#include"JacobiGpu.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <time.h>
#include <string>
#include <iostream>
#include <fstream>
#include "GpuHelper.cuh"
#include "Util.h"
using namespace std;

JacobiGpu::JacobiGpu(int dimension) {

	this->size = dimension;
	N = size * size;

	A = (double *)malloc(N * sizeof(double));
	x = (double *)malloc(size * sizeof(double));
	b = (double *)malloc(size * sizeof(double));
	nextX = (double *)malloc(size * sizeof(double));

	for (int i = 0; i < size; i++) {
		x[i] = 0;
		nextX[i] = 0;
	}

	GpuHelper * gpuHelper = new GpuHelper();
	int devId = 0, devsNum = 1;
	gpuHelper->selectGpu(&devsNum, &devId);
	gpuHelper->testDevice(devId);

	assert(cudaSuccess == cudaMalloc((void **)&dNextX, size * sizeof(double)));
	assert(cudaSuccess == cudaMalloc((void **)&dA, N * sizeof(double)));
	assert(cudaSuccess == cudaMalloc((void **)&dX, size * sizeof(double)));
	assert(cudaSuccess == cudaMalloc((void **)&dB, size * sizeof(double)));
}

void JacobiGpu::solve(double eps) {

	int iter = maxIterations;
	int rowSize = size, colSize = size;

	CpuTimer timer = CpuTimer();

	cudaMemcpy(dNextX, nextX, sizeof(double)*rowSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dA, A, sizeof(double)*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dX, x, sizeof(double)*rowSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, b, sizeof(double)*rowSize, cudaMemcpyHostToDevice);

	double diff = 0.1;
	int k = 0;
	double tempDiff = 0;

	//float elapsedTime = 0.0;
	//cudaEvent_t event_start, event_stop;
	//cudaEventCreate(&event_start);
	//cudaEventCreate(&event_stop);
	//cudaEventRecord(event_start, 0);

	int multicity = int(0.1 / eps);
	timer.start();

	if (type == 1) {
		int blockSize = rowSize;
		int nBlocks = 1;
		int nTiles = rowSize / TileSize + (rowSize%TileSize == 0 ? 0 : 1);			//gride中block 数目，每个block中的Thread数量
		for (; k < iter&& diff >eps; k++) {
			diff = 0;
			// if (k % 2)
			// 	jacobiIteration << < nTiles, TileSize>> > (dX, dA, dNextX, dB, rowSize, colSize);// ,rowSize*sizeof(double)
			// else
			// 	jacobiIteration << < nTiles, TileSize>> > (dNextX, dA, dX, dB, rowSize, colSize);
			cudaDeviceSynchronize();
			cudaMemcpy(nextX, dNextX, sizeof(double)*rowSize, cudaMemcpyDeviceToHost);
			cudaMemcpy(x, dX, sizeof(double)*rowSize, cudaMemcpyDeviceToHost);

			for (int i = 0; i < rowSize; i++) {
				tempDiff = fabs(x[i] - nextX[i]);
				if (tempDiff > diff && tempDiff != 0) {
					diff = tempDiff;
				}
			}
			if (diff < eps*multicity) {
				std::cout << timer.stop() << " ";
				multicity = int(multicity / 10);
			}
		}
		iter = k;
	}
	else {
		for (k = 0; k < iter && diff > eps; k++) {
			diff = 0;
			if (k % 2)
				jacobiIterationWithSharedMemory << < rowSize, TileSize >> > (dX, dA, dNextX, dB, rowSize, colSize);
			else
				jacobiIterationWithSharedMemory << < rowSize, TileSize >> > (dNextX, dA, dX, dB, rowSize, colSize);

			cudaDeviceSynchronize();
			cudaMemcpy(nextX, dNextX, sizeof(double)*rowSize, cudaMemcpyDeviceToHost);
			cudaMemcpy(x, dX, sizeof(double)*rowSize, cudaMemcpyDeviceToHost);
			for (int i = 0; i < rowSize; i++) {
				tempDiff = fabs(x[i] - nextX[i]);
				if (tempDiff > diff && tempDiff != 0) {
					diff = tempDiff;
				}
			}
			if (diff < eps*multicity) {
				std::cout << timer.stop() << " ";
				multicity = int(multicity / 10);
			}
		}
		iter = k;
	}

	//cudaEventRecord(event_stop, 0);
	//cudaEventSynchronize(event_stop);
	//cudaEventElapsedTime(&elapsedTime, event_start, event_stop);
	//cudaDeviceSynchronize();

	std::cout << std::endl << "Iterations:" << iter <<std::endl;

}

void JacobiGpu::freeAllMemory() {
	free(nextX); free(A); free(x); free(b);
	cudaFree(dNextX); cudaFree(dA); cudaFree(dX); cudaFree(dB);
}

void JacobiGpu::input(std::string rFile) {
	int n = size;
	ifstream  fin(rFile);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			fin >> A[i*n + j];
			//cout << A[i][j] << " ";
		}
		//cout << endl;
	}
	for (int i = 0; i < n; i++) {
		fin >> b[i];
	}
	fin.close();
}

void JacobiGpu::output(std::string wfile) {
	int n = size;
	ofstream fout(wfile);
	for (int i = 0; i < n; i++)
	{
		fout << x[i] << "   ";
		//cout << x[i] << "   ";
	}
	fout.close();
}

__global__ void jacobiIterationWithSharedMemory(double* nextX, const double* __restrict__ A, const double* __restrict__  x, const double* __restrict__ b, int rowSize, int colSize) {

	int col = threadIdx.x;
	int row = blockIdx.x;
	int offset = row * colSize;
	int wid = threadIdx.x & 0x1f;				 //同一warp内线程的id

	double sum = 0.0;
	//__shared__ double tmpresult[8];

	if (row < rowSize) {
		for (int i = col; i < colSize; i += blockDim.x)
			sum += A[offset + i] * x[i];

		sum += __shfl_down_sync(__activemask(), sum, 16);  //先进行warp内的规约
		sum += __shfl_down_sync(__activemask(), sum, 8);
		sum += __shfl_down_sync(__activemask(), sum, 4);
		sum += __shfl_down_sync(__activemask(), sum, 2);
		sum += __shfl_down_sync(__activemask(), sum, 1);
		//__activemask返回一个32位无符号整形掩码（mask），该掩码的每一位表示了当前warp中线程的活动（active）情况。换句话说，该掩码表示了在同一个warp中也在执行__active_mask()的线程
		// if(wid==0)
		// 	tmpresult[threadIdx.x>>5]=sum;                    //Block内的每个warp的第一个线程写入共享内存
		// __syncthreads();

		// if(threadIdx.x<8){
		// 	sum = tmpresult[wid];
		//     sum += __shfl_down_sync(__activemask(),sum, 4);
		//     sum += __shfl_down_sync(__activemask(),sum, 2);
		//     sum += __shfl_down_sync(__activemask(),sum, 1);
		// }
		if (threadIdx.x == 0)
			nextX[row] = (b[row] - (sum - A[offset + row] * x[row])) / A[offset + row];                    //N个block分别把第一个线程内的结果写入数组对应位置
	}
}

// __global__ void JacobiGpu::jacobiIteration(double* nextX, double* A, double* x, double* b, int rowSize, int colSize){
// 	int idx = blockIdx.x*blockDim.x + threadIdx.x;
// 	if (idx < rowSize){
// 		double sum = 0.0;
// 		int idxForARowI = idx * colSize;
// 		for (int j = 0; j < colSize; j++)
// 			sum += A[idxForARowI + j] * x[j];
// 		sum = sum - A[idxForARowI + idx] * x[idx];
// 		nextX[idx] = (b[idx] - sum) / A[idxForARowI + idx];
// 	}
// }

int main(int argc, char ** argv) {
	int dimension = stoi(argv[1], 0, 10);
	JacobiGpu * jacobi = new JacobiGpu(dimension);
	jacobi->input(argv[2]);
	double eps = stod(argv[3]);
	jacobi->solve(eps);
	jacobi->output(argv[4]);
	jacobi->freeAllMemory();
}

