#include<ctime>
#ifndef UTIL_H
#define UTIL_H
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
#endif