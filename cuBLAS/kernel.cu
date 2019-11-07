
// CUDA runtime 库 + CUBLAS 库
#include <cublas_v2.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"device_functions.h"

#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include<iostream>
using namespace std;

using namespace std;

// 定义测试矩阵的维度
int  sizeT = 4096;
int  aW = sizeT;
int  aH = sizeT;
int  bW = sizeT;
int  bH = sizeT;
int  cW = bW;
int  cH = aH;
void mul_cpu(float * a, float * b, float * c) {
	for (int i = 0; i < aH; i++) {
		for (int j = 0; j < bW; j++) {
			c[i*bW + j] = 0;
			for (int k = 0; k < aW; k++) {
				c[i*bW + j] += (a[i*aW + k]) * (b[k*bW + j]);
			}
		}
	}
}
float check_diff(float * gpuC, float * cpuC) {
	float diff = 0;
	for (int i = 0; i < aH*bW; i++) {
		diff += abs(gpuC[i] - cpuC[i]);
	}
	return diff;
}

int matrix()
{
	// 定义状态变量
	cublasStatus_t status;

	// 在 内存 中为将要计算的矩阵开辟空间
	float *h_A = (float*)malloc(aW*aH * sizeof(float));
	float *h_B = (float*)malloc(bW*bH * sizeof(float));

	// 在 内存 中为将要存放运算结果的矩阵开辟空间
	float *h_C = (float*)malloc(cW*cH * sizeof(float));

	// 在 内存 中为将要存放运算结果的矩阵开辟空间
	float *cpu_C = (float*)malloc(cW*cH * sizeof(float));

	//为待运算矩阵的元素赋值
	for (int i = 0; i < aW*aH; ++i)
	{
		h_A[i] = (float)(rand() / (float)RAND_MAX);//(float)(rand() % 10);// (float)(rand() / (float)RAND_MAX); //(float)(rand() %10); (float)i;
	}
	for (int i = 0; i < bW*bH; ++i)
	{
		h_B[i] = (float)(rand() / (float)RAND_MAX);//(float)(rand() % 10);//  //(float)i*i;
	}
	//h_A[0] = 1; h_A[1] = 1;
	//h_B[0] = 2; h_B[1] = 1; h_B[2] = 0; h_B[3] = 1; h_B[4] = 3; h_B[5] = 0;



	//// 打印待测试的矩阵
	//cout << "矩阵 A :" << endl;
	//for (int i = 0; i < aW*aH; i++) {
	//	cout << h_A[i] << " ";
	//	if ((i + 1) % aW == 0) cout << endl;
	//}
	//cout << endl;
	//cout << "矩阵 B :" << endl;
	//for (int i = 0; i < bW*bH; i++) {
	//	cout << h_B[i] << " ";
	//	if ((i + 1) % bW == 0) cout << endl;
	//}
	//cout << endl;

	/*
	** GPU 计算矩阵相乘
	*/

	
	float elapsedTime = 0.0;
	cudaEvent_t event_start, event_stop;
	cudaEventCreate(&event_start);
	cudaEventCreate(&event_stop);
	cudaEventRecord(event_start, 0);

	// 创建并初始化 CUBLAS 库对象
	cublasHandle_t handle;
	status = cublasCreate(&handle);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
			cout << "CUBLAS 对象实例化出错" << endl;
		}
		getchar();
		return EXIT_FAILURE;
	}



	float *d_A, *d_B, *d_C;
	// 在 显存 中为将要计算的矩阵开辟空间
	cudaMalloc(
		(void**)&d_A,    // 指向开辟的空间的指针
		aW*aH * sizeof(float)    //　需要开辟空间的字节数
	);
	cudaMalloc(
		(void**)&d_B,
		bW*bH * sizeof(float)
	);

	// 在 显存 中为将要存放运算结果的矩阵开辟空间
	cudaMalloc(
		(void**)&d_C,
		cW*cH * sizeof(float)
	);

	// 将矩阵数据传递进 显存 中已经开辟好了的空间
	cublasSetVector(
		aW*aH,    // 要存入显存的元素个数
		sizeof(float),    // 每个元素大小
		h_A,    // 主机端起始地址
		1,    // 连续元素之间的存储间隔
		d_A,    // GPU 端起始地址
		1    // 连续元素之间的存储间隔
	);
	cublasSetVector(
		bW*bH,
		sizeof(float),
		h_B,
		1,
		d_B,
		1
	);

	// 同步函数
	cudaThreadSynchronize();

	// 传递进矩阵相乘函数中的参数，具体含义请参考函数手册。
	float a = 1; float b = 0;
	// 矩阵相乘。该函数必然将数组解析成列优先数组
	cublasSgemm(
		handle,    // blas 库对象
		CUBLAS_OP_N,    // 矩阵 A d_B不转置
		CUBLAS_OP_N,    // 矩阵 B  d_A不转置
		bW,    // d_B, C 的行数
		aH,    // d_A, C 的列数
		bH,    // d_B 的列数和 d_A 的行数
		&a,    // 运算式的 α 值 1
		d_B,    // A 在显存中的地址
		bW,    // lda  使d_B转置
		d_A,    // B 在显存中的地址
		aW,    // ldb 使d_A转置
		&b,    // 运算式的 β 值
		d_C,    // C 在显存中的地址(结果矩阵)
		cW    // ldc
	);
	// 同步函数
	cudaThreadSynchronize();

	// 从 显存 中取出运算结果至 内存中去
	cublasGetVector(
		cW*cH,    //  要取出元素的个数
		sizeof(float),    // 每个元素大小
		d_C,    // GPU 端起始地址
		1,    // 连续元素之间的存储间隔
		h_C,    // 主机端起始地址
		1    // 连续元素之间的存储间隔
	);
	cudaEventRecord(event_stop, 0);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&elapsedTime, event_start, event_stop);
	// 同步device 保证结果能正确访问
	cudaDeviceSynchronize();
	printf("matrix size:%d * %d \n", sizeT,sizeT);
	printf("cuda event time = %lfms\n", elapsedTime);
	mul_cpu(h_A, h_B, cpu_C);

	// 打印运算结果
	//cout << "计算结果的转置 ( (A*B)的转置 )：" << endl;

	//for (int i = 0; i < cW*cH; i++) {
	//	cout << h_C[i] << " ";
	//	if ((i + 1) % cW == 0) cout << endl;
	//}

	//
	//cout << "cpu计算结果：" << endl;

	//for (int i = 0; i < cW*cH; i++) {
	//	cout << cpu_C[i] << " ";
	//	if ((i + 1) % cW == 0) cout << endl;
	//}
	float diff = check_diff(h_C, cpu_C);
	printf("diff is %.10f\n", diff);
	// 清理掉使用过的内存
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// 释放 CUBLAS 库对象
	cublasDestroy(handle);
	return 0;
}

int main() {
	for (int i = 0; i < 10; i++) {
		int t = pow(2, i);
		sizeT = t;
		aW = sizeT;
		aH = sizeT;
		bW = sizeT;
		bH = sizeT;
		cW = bW;
		cH = aH;
		matrix();
		
	}
	return 0;
}