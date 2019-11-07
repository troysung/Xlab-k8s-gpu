#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"device_functions.h"
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include<iostream>
using namespace std;
#define BLOCK_SIZE 32
// 矩阵以行为主索引
// 即M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
	int width;
	int height;
	// 步长 方便索引
	int stride;
	// 该矩阵第一个元素的指针
	float* elements;
} Matrix;

// 根据row和col获得对应矩阵中元素的值
__device__ float GetElement(const Matrix A, int row, int col)
{
	return A.elements[row * A.stride + col];
}

// 设置 (row,col)位置的值
__device__ void SetElement(Matrix A, int row, int col,
	float value)
{
	A.elements[row * A.stride + col] = value;
}

// 获得子矩阵
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	// A.stride * BLOCK_SIZE * row 指的是前row-1行个block的元素个数
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
		+ BLOCK_SIZE * col];
	return Asub;
}

// 矩阵乘法内核函数的前向声明
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// 矩阵乘法的Host部分
// 矩阵的尺寸假定为BLOCK_SIZE的倍数
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	//将cpu上的矩阵AB拷贝到GPU上
	Matrix d_A;
	d_A.width = d_A.stride = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	//在CUDA的全局内存上分配d_A矩阵的空间
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size,
		cudaMemcpyHostToDevice);
	Matrix d_B;
	d_B.width = d_B.stride = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	//在CUDA的全局内存上分配d_B矩阵的空间
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size,
		cudaMemcpyHostToDevice);

	// 在CUDA的全局内存分配结果矩阵d_C的空间
	Matrix d_C;
	d_C.width = d_C.stride = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);

	// 唤醒内核
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((B.width+dimBlock.x-1)/ dimBlock.x, (A.height + dimBlock.y -1) / dimBlock.y);
	// 设置内核函数的线程格 和 线程块大小，并运行
	MatMulKernel << <dimGrid, dimBlock >> >(d_A, d_B, d_C);

	// 将d_C从GPU内存拷贝到cpu中
	cudaMemcpy(C.elements, d_C.elements, size,
		cudaMemcpyDeviceToHost);

	// 释放之前在cuda的全局内存上分配的空间
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// 矩阵乘法内核函数
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	//获得当前线程所在块的块索引
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// 每个线程块计算结果矩阵的一个子矩阵
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

	// 用来记录Csub（row,col）位置的值
	float Cvalue = 0;

	// 获得当前线程在块内的索引
	int row = threadIdx.y;
	int col = threadIdx.x;

	//循环计算Csub所需的所有A和B子矩阵将每对子矩阵相乘并累加结果
	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

		// 获得A的子矩阵
		Matrix Asub = GetSubMatrix(A, blockRow, m);

		// 获得B的子矩阵
		Matrix Bsub = GetSubMatrix(B, m, blockCol);

		// 使用共享内存来分别存储 Asub and Bsub ，共享内存可被一个线程块内所有线程访问
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// 将Asub和Bsub从全局内存加载到共享内存中，其中每个线程加载每个子矩阵的一个元素
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);

		// 在开始计算之前进行同步来确保子矩阵都加载到共享内存中
		__syncthreads();

		// 将Asub的一行乘BSub的一列
		for (int e = 0; e < BLOCK_SIZE; ++e)
			Cvalue += As[row][e] * Bs[e][col];

		// 进行同步以确保Cvalue的计算，在下一次迭代中前完成（加载两个新的子矩阵）
		__syncthreads();
	}

	// 将Cvalue写入Csub中
	// 每个线程仅写一个元素
	SetElement(Csub, row, col, Cvalue);
}
// 用循环遍历的方式在cpu上计算矩阵乘法，复杂度n^3
void mul_cpu(Matrix a, Matrix b,Matrix c) {
	int size = a.width;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			c.elements[i * c.stride + j] = 0;
			for (int k = 0; k < size; k++) {
				c.elements[i * c.stride + j] += (*(a.elements + i * a.stride + k)) * (*(b.elements + k * b.stride + j));
			}
		}
	}
}
// 计算cpu和gpu上矩阵乘法的差值
float check_diff(Matrix c,Matrix C,int size) {
	float diff = 0;
	for (int i = 0; i < size*size; i++) {
		diff += abs(c.elements[i] - C.elements[i]);
	}
	return diff;
}
int test(int size);
int main() {
	for (int i = 0; i <= 10; i++) {
		
		int size = pow(2, i);
		cout << size << endl;
		test(size);
	}
}
int test(int size)
{
	// 初始化矩阵
	int width = size;
	int height = size;
	Matrix A;
	Matrix B;
	Matrix C;
	Matrix c;
	A.width = width; A.height = height;
	B.width = width; B.height = height;
	A.stride = width; B.stride = width;
	B.width = width; B.height = height;
	C.width = height; C.height = width;
	c.width = height; c.height = width;
	c.stride = width; C.stride = width;
	int nBytes = width * height * sizeof(float);
	A.elements = (float*)malloc(nBytes);
	B.elements = (float*)malloc(nBytes);
	C.elements = (float*)malloc(nBytes);
	c.elements = (float*)malloc(nBytes);
	// ��ʼ������
	for (int i = 0; i < width * height; ++i)
	{
		A.elements[i] = (float)(rand() / (float)RAND_MAX);;
		B.elements[i] = (float)(rand() / (float)RAND_MAX);;
	}
	//启动一个cudaEvent用来给cuda矩阵乘法计时
	float elapsedTime = 0.0;
	cudaEvent_t event_start, event_stop;
	cudaEventCreate(&event_start);
	cudaEventCreate(&event_stop);
	cudaEventRecord(event_start, 0);

	MatMul(A, B, C);
	
	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&elapsedTime, event_start, event_stop);
	// 同步device 保证结果能正确访问
	cudaDeviceSynchronize();
	printf("cuda event time = %lfms\n", elapsedTime);
	
	//启动一个clock给cpu矩阵乘法计时
	double start = 0.0f, end = 0.0f;
	clock_t clock_start;
	clock_t clock_end;
	clock_start = clock();

	mul_cpu(A, B, c);

	clock_end = clock();
	double clock_diff_sec = ((double)(clock_end - clock_start) / CLOCKS_PER_SEC);
	
	float diff = check_diff(c, C, C.width);
	printf("diff is %.10f\n", diff);
	
	printf("cpu cal clock_ time: %lfms.\n", clock_diff_sec * 1000);

	// ͬ��device ��֤�������ȷ����
	// ���ִ�н��
	/*cudaDeviceReset()*/;

	return 0;
}
