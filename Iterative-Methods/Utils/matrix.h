#include <iostream>
#include<stdio.h>
#include<math.h>
#include<random>
#include <iomanip>
#include<ctime>
using namespace std;
#ifndef MATRIX_H
#define MATRIX_H	
extern void setRandomValueForVector(double * m, int size);

extern void setRandomValueForMatrix(double **m, int row, int col);

template<class T>
T ** createMatrix(int row, int col)
{
	T ** matrix = new T*[row];
	for (int i = 0; i < row; i++)
	{
		matrix[i] = new T[col];
	}
	return matrix;
}

template<class T>
T * createVector(int size)
{
	T * vector = new T[size];
	return vector;
}

template<class T>
void freeMatrix(T **m, int size) {
	for (int i = 0; i < size; i++)
		delete[]m[i];
	delete[]m;
}

//n*m m*p
template <class T>
void matrixMultiply(T ** a, T * b, T *c, int n, int m, int p)
{
	for (int i = 0; i < n; i++)
	{
		T sum = 0;
		for (int j = 0; j < m; j++)
		{
			sum += a[i][j] * b[j];
		}
		c[i] = sum;
	}
}

template <class T>
T checkDiff(T * x, T * result, int size)
{
	T sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum += abs(x[i] - result[i]);
	}
	return sum;
}

template <class T>
void show_matrix(T ** matrix, int matrixSize)
{
	for (int i = 0; i < matrixSize; i++)
	{
		for (int j = 0; j < matrixSize; j++)
		{
			cout << matrix[i][j] << "\t";
		}

		cout << endl;
	}
	cout << endl;
}

template <class T>
void show_vector(T *x, int size)
{
	for (int i = 0; i < size; i++)
	{
		cout << setprecision(12) << x[i] << "\t";
	}
	cout << endl;
}
#endif