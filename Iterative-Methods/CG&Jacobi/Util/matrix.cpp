#include"matrix.h"
#include <iostream>
#include<stdio.h>
#include<math.h>
#include<random>
#include <iomanip>
#include<ctime>
using namespace std;
void setRandomValueForVector(double * m, int size) {
	//以当前时间为随机种子
	default_random_engine e(time(0));
	uniform_real_distribution<double > uRD(-10000,10000);
	for (int k = 0; k < size; k++) {
		m[k] = uRD(e);
	}
}

void setRandomValueForMatrix(double **m, int row, int col) {
	//以当前时间为随机种子
	default_random_engine e(time(0));
	uniform_real_distribution<double > uRD(-10000, 10000);
	for (int k = 0; k < row; k++) {
		double sum = 0;
		for (int j = 0; j < col; j++) {
			m[k][j] = uRD(e);
			sum += abs(m[k][j]);
		}
		m[k][k] = sum + abs(uRD(e));
	}
}


