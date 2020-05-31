#ifndef JACOBICPU_H
#define JACOBICPU_H
#include <string>
#include"BasicSolver.h"
class JacobiCpu:public BasicSolver {
public:
	JacobiCpu(int dimension) :BasicSolver(dimension) {};
	void freeAllMemory();
	void solve(double eps);
};
#endif

