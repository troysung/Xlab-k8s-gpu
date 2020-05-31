#include"SparseSolver.h"
#include <stdio.h>
#include <stdlib.h>
#define BAISC 0
#define UM 1
#define CUDAGRAPH 2
#define TEST 3
void test(int argc, char **argv, int size, std::string rfile) {
	std::string wFile = "test.txt";
	printf("===================warm up====================\n");
	conjugateGradient(argc, argv, size, rfile, 1, wFile);
	printf("===================warm up====================\n");
	for (float eps = 1e-7; eps >= 1e-10; eps /= 10.0) {
		printf("===================eps %.10f=======================\n",eps);
		//conjugateGradient(argc, argv, size, rfile, eps, wFile);
		conjugateGradientUM(argc, argv, size, rfile, eps, wFile);
		//conjugateGradientCudaGraphs(argc, argv, size, rfile, eps, wFile); 
	}
}
int main(int argc, char **argv) {
	int type = std::stoi(argv[1]);
	int dimension = std::stoi(argv[2], 0, 10);
	std::string rFile = argv[3];
	std::string wFile = argv[5];
	float eps = std::stod(argv[4]);
	switch (type)
	{
	case BAISC:
		conjugateGradient(argc, argv, dimension, rFile, eps, wFile);
		break;
	case UM:
		conjugateGradientUM(argc, argv, dimension, rFile, eps, wFile);
		break;
	case CUDAGRAPH:
		conjugateGradientCudaGraphs(argc, argv, dimension, rFile, eps, wFile);
		break;
	case TEST:
		test(argc, argv, dimension, rFile);
		break;
	default:
		printf("method type error!");
		break;
	}	
	return 0;
}