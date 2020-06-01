#include"BasicSolver.h"
#include"matrix.h"
#include  <fstream>
BasicSolver::BasicSolver(int dimension) {
	this->size = dimension;

	this->A = createMatrix<double>(size, size);

	this->b = createVector<double>(size);

	this->x = createVector<double>(size);

	this->nextX = createVector<double>(size);

	for (int i = 0; i < size; i++) {
		x[i] = 0;
	}
}

void BasicSolver::input(string rFile) {
	int n = this->size;
	ifstream  fin(rFile);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			fin >> A[i][j];
			//cout << A[i][j] << " ";
		}
		//cout << endl;
	}
	for (int i = 0; i < n; i++) {
		fin >> b[i];
	}
	fin.close();
}

void BasicSolver::output(string wfile) {
	int n = size;
	ofstream fout(wfile);
	for (int i = 0; i < n; i++)
	{
		fout << x[i] << "   ";
		//cout << x[i] << "   ";
	}
	fout.close();
}

void BasicSolver::freeMemory() {
	freeMatrix(A, size);
	delete[]b;
	delete[]x;
	delete[]nextX;
}