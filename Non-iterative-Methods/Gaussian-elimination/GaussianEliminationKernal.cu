#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "GaussianElimination.h"

__global__ void gauss_eliminate_kernel(float * U, int ops_per_thread)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i, j, k;
	unsigned int num_rows = MATRIX_SIZE;
	
	//Contents of the A matrix should already be in U
	for(k = 0; k < num_rows; k++)
	{
		if(tx==0)
		{		
			for(j = (k + 1); j < num_rows; j++)
			{
				U[k * num_rows + j] /= U[k * num_rows + k]; // Division step
			}
		 U[k * num_rows + k] =1;
		}
		
		//Sync threads(only one thread block so, ok)
		__syncthreads();

		//Elimination step
		int itop = num_rows-1;
		//Bottom limit on i for whole (original) loop
		int ibottom = k+1; 
		
		//Each thread does so many iterations of elimination step
		//Starting index for this thread
		int istart = tx*ops_per_thread + ibottom;
		//Ending index for this thread
		int iend = (istart + ops_per_thread)-1;
		
		//Check boundaries, else do nothing
		if( (istart >= ibottom) && (iend <= itop))
		{
			for(i = istart; i <= iend; i++)
			{
				//Do work  for this i iteration
				for(j = i; j < num_rows; j++)
				{
					U[i * num_rows + j] -= U[i * num_rows + k] * U[k * num_rows + j];

				}
			}
		}
	
		__syncthreads();
	}

	__syncthreads();
	
	int itop = num_rows-1;
	//Bottom limit on i for whole (original) loop
	int ibottom = 0;
	
	int istart = tx*ops_per_thread + ibottom;
	//Ending index for this thread
	int iend = (istart + ops_per_thread)-1;
	
	//Check boundaries, else do nothing
	if( (istart >= ibottom) && (iend <= itop))
	{
		for(i = istart; i <= iend; i++)
		{
			//Do work  for this i iteration
			for(j = 0; j < i; j++)
			{
				U[i * num_rows + j] = 0.0;
			}
		}
	}

}

__global__ void gauss_eliminate_kernel_optimized_div(float * U, int k, int stride)
{	
	//General thread id
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	
	//Iterators
	unsigned int j;
	unsigned int num_rows = MATRIX_SIZE;

	int offset = (k+1); //From original loop
	int jstart = threadIdx.x + offset;
	int jstep = stride;

	//Only continue if in bounds?
	//Top limit on i for whole (original) loop
	int jtop = num_rows-1;
	//Bottom limit on i for whole (original) loop
	int jbottom = (k + 1);
	
	//Do work for this i iteration
	//Division step
	//Only let one thread block do this
	if(blockIdx.x == 0)
	{
		for(j = jstart; (j >= jbottom) && (j <= jtop); j+=jstep)
		{
			U[k * num_rows + j] /= U[k * num_rows + k]; // Division step
		}
	 U[k * num_rows + k] =1;

	}
}

__global__ void gauss_eliminate_kernel_optimized(float * U, int k, int stride)
{

	unsigned int j;
	unsigned int num_rows = MATRIX_SIZE;
	
	
	int i = blockIdx.x + (k+1);
	int offset = i; //From original loop
	int jstart = threadIdx.x + offset;
	int jstep = stride;
	int jtop = num_rows-1;
	int jbottom = i; 
	
	for(j = jstart; (j >= jbottom) && (j <= jtop); j+=jstep)
	{
		U[i * num_rows + j] -= U[i * num_rows + k] * U[k * num_rows + j];
	}
}


#endif // #ifndef _MATRIXMUL_KERNEL_H_