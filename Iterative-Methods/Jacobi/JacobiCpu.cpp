#include"JacobiCpu.h"
#include "Util.h"
#include  <fstream>
#include<iostream>
using namespace std;


void JacobiCpu::freeAllMemory() {
	freeMemory();
}
void JacobiCpu::solve(double eps) {

	CpuTimer timer = CpuTimer();
	timer.start();
	
	 //最大迭代次数
	double residual = 0.0;  //	
	double sum = 0.0;
	double dis = 0.0;
	double diff = 1.0;  //相邻迭代的结果差
	int multicity = int(0.1 / eps);
	timer.start();

	int count = 1;
	for (; (count < maxIterations) && (diff > eps); count++)
	{
		diff = 0.0;

		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				if (i != j)
				{
					sum += A[i][j] * x[j];
				}
			}
			nextX[i] = (b[i] - sum) / A[i][i];
			sum = 0.0;
		}
		residual = 0.0;
		//计算相邻迭代的结果差
		for (int m = 0; m < size; m++)
		{
			dis = fabs(nextX[m] - x[m]);
			if (dis > residual)
				residual = dis;
		}
		diff = residual;
		if (diff < eps*multicity) {
			cout << timer.stop() << " ";
			multicity = int(multicity / 10);
		}
		memcpy(x, nextX, size * sizeof(double));
	}

	cout << endl << "Iterations:" << count << endl;
}

int main(int argc, char ** argv) {
	int dimension = stoi(argv[1], 0, 10);
	//cout << dimension;
	JacobiCpu * jacobi = new JacobiCpu(dimension);
	jacobi->input(argv[2]);
	double eps = stod(argv[3]);
	jacobi->solve(eps);
	jacobi->output(argv[4]);
	jacobi->freeAllMemory();
}