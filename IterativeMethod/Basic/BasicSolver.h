#ifndef BASICSOLVER_H_
#define BASICSOLVER_H_
#include <string>
class BasicSolver {
protected:
	const int maxIterations = 40000;
	double **A;
	double * b;
	double * x;
	double * nextX;
	int size;
	void freeMemory();
public:
	BasicSolver(int dimension);
	void input(std::string rFile);
	void output(std::string wfile);
	void virtual solve(double eps) = 0;
};
#endif