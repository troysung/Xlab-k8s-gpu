#ifndef CONJUGATEGRADIENTCPU_H_
#define CONJUGATEGRADIENTCPU_H_
#define b(x) (b[(x)])
#include<string>
class ConjugateGradientCpu{
private:
	float * A;
	float * b;
	float * x;
	int size;
	const int maxIterations = 1000;
public:
	ConjugateGradientCpu(int dimension);
	void solve(double eps);
	void input(std::string rFile);
	void output(std::string wfile);
	void freeAllMemory();
};
#endif 
