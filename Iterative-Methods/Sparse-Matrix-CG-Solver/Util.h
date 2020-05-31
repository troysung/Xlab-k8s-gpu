#ifndef UTIL_H_
#define UTIL_H_

#include<fstream>
#include <string>
#include<ctime>
class CpuTimer {
	clock_t clock_start;
	clock_t clock_end;
public:
	CpuTimer() {

	}
	void start() {
		clock_start = clock();
	}
	double stop() {
		clock_end = clock();
		double clock_diff_sec = ((double)(clock_end - clock_start));
		return clock_diff_sec;
	}
};
extern void input(int *I, int *J, float *val, std::string rFile, int nz, int N);

/* genTridiag: generate a random tridiagonal symmetric matrix */
extern void genTridiag(int *I, int *J, float *val, int N, int nz);

extern void output(std::string wfile, float *x, int size);

#endif