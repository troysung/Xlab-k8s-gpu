#ifndef GPUHELPER_CUH
#define GPUHELPER_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class GpuHelper{
public:
	static void selectGpu(int *devsNum, int *gpuNum);
	static void testDevice(int devId);
};

#endif