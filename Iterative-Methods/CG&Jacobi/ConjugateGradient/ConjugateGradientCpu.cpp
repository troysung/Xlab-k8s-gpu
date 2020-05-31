// Sequential version of CG
// Author: Tim Lebailly

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include"Util.h"
#include"ConjugateGradientCpu.h"
#include  <fstream>
#include<iostream>
#define TOL 0.001
using namespace std;

/*
 * Computes a (square) matrix vector product
 * Input: pointer to 1D-array-stored matrix (row major), 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
void matVec(float* A, float* b, float* out, int size) {
	int i, j;
	for (j = 0; j < size; j++) {
		out[j] = 0;
		for (i = 0; i < size; i++) {
			out[j] += A[j*size + i] * b(i);
		}
	}
}

/*
 * Computes the scalar product of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer 1D-array-stored vector
 * Output: float scalar product
 */
float vecVec(float* vec1, float* vec2, int size) {
	int i;
	float product = 0;
	for (i = 0; i < size; i++) {
		product += vec1[i] * vec2[i];
	}
	return product;
}

/*
 * Computes the sum of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the sum in memory at the location of the pointer out
 */
void vecPlusVec(float* vec1, float* vec2, float* out, int size) {
	int i;
	for (i = 0; i < size; i++) {
		out[i] = vec1[i] + vec2[i];
	}
}

/*
 * Computes a scalar vector product
 * Input: scalar, pointer to 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
void scalarVec(float alpha, float* vec2, float* out, int size) {
	int i;
	for (i = 0; i < size; i++) {
		out[i] = alpha * vec2[i];
	}
}

/*
 * Computes a scalar (square) matrix vector product
 * Input: scalar, pointer to 1D-array-stored matrix (row major), pointer to 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
void scalarMatVec(float alpha, float* A, float* b, float* out, int size) {
	int i, j;
	for (j = 0; j < size; j++) {
		out[j] = 0;
		for (i = 0; i < size; i++) {
			out[j] += alpha * A[j*size, i] * b(i);
		}
	}
}

/*
 * Computes the 2-norm of a vector
 * Input: pointer to 1D-array-stored vector
 * Output: value of the norm of the vector
 */
float norm2d(float* a, int size) {
	return sqrt(vecVec(a, a, size));
}

ConjugateGradientCpu::ConjugateGradientCpu(int dimension) {
	this->size = dimension;
	A = (float *)malloc(sizeof(float) * size * size);
	b = (float *)malloc(sizeof(float) * size);
	x = (float *)malloc(size * sizeof(float));
}

void ConjugateGradientCpu::freeAllMemory() {
	free(A);
	free(b);
	free(x);
}


void ConjugateGradientCpu::input(std::string rFile) {
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
		x[i] = 0;
	}
	fin.close();
}
void ConjugateGradientCpu::output(std::string wfile) {
	int n = size;
	ofstream fout(wfile);
	for (int i = 0; i < n; i++)
	{
		fout << x[i] << "   ";
		//cout << x[i] << "   ";
	}
	fout.close();
}
/*
 * Solve the system Ax=b using the CG method
 * Input: pointer to 1D-array-stored matrix, pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * float* x is used as initial condition and the final output is written there as well
 */
void ConjugateGradientCpu::solve(double eps) {
	// Initialize temporary variables
	float* p = (float*)calloc(sizeof(float), size);
	float* r = (float*)calloc(sizeof(float), size);
	float* temp = (float*)calloc(sizeof(float), size);
	float beta, alpha, rNormOld = 0.0;
	float rNorm = 1.0;
	int k = 0;

	// Set initial variables
	scalarMatVec(-1.0, A, x, temp, size);
	vecPlusVec(b, temp, r, size);
	scalarVec(1.0, r, p, size);
	rNormOld = vecVec(r, r, size);

	CpuTimer timer = CpuTimer();
	int multicity = int(0.1 / eps);
	timer.start();
	while ((rNorm > eps) && (k < maxIterations)) {
		// temp = A* p (only compute matrix vector product once)
		matVec(A, p, temp, size);
		// alpha_k = ...
		alpha = rNormOld / vecVec(p, temp, size);
		// r_{k+1} = ...
		scalarVec(-alpha, temp, temp, size);
		vecPlusVec(r, temp, r, size);
		// x_{k+1} = ...
		scalarVec(alpha, p, temp, size);
		vecPlusVec(x, temp, x, size);
		// beta_k = ...
		rNorm = vecVec(r, r, size);
		beta = rNorm / rNormOld;
		// p_{k+1} = ...
		scalarVec(beta, p, temp, size);
		vecPlusVec(r, temp, p, size);
		// set rOld to r
		rNormOld = rNorm;
		if (rNorm < eps*multicity) {
			std::cout << timer.stop() << " ";
			multicity = int(multicity / 10);
		}
		k++;
	}
	std::cout << std::endl << "Iterations:" << k << std::endl;
	// free temporary memory
	free(p);
	free(r);
	free(temp);
}
void main(int argc, char ** argv) {
	int dimension = stoi(argv[1], 0, 10);
	ConjugateGradientCpu * cGC = new ConjugateGradientCpu(dimension);
	cGC->input(argv[2]);
	double eps = stod(argv[3]);
	cGC->solve(eps);
	cGC->output(argv[4]);
	cGC->freeAllMemory();
}
