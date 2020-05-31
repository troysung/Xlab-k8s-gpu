#ifndef CAL_X
#define CAL_X

#include "reduction_aux.h"
#include <assert.h>



/* partial dot product */
__global__ void dot_stg_1_x(const FLOAT *u, FLOAT *x, FLOAT *z, int N, int row)
{
    __shared__ FLOAT sdata[256];
    int endPos = row * N + N;
    int xInd = row + 1 + get_tid();
    int idx = row * N + xInd;
    int tid = threadIdx.x;
    int bid = get_bid();

    /* load data to shared mem */
    if (idx < endPos) {
        sdata[tid] = u[idx] * x[xInd];
        //printf("%d ,u %f * x %f = %f\n",tid,u[idx],x[xInd],sdata[tid]);
    }
    else {
        sdata[tid] = 0;//avoid warp divergence in following reduction
    }

    __syncthreads();

    /* reduction using shared mem */
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();

    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) z[bid] = sdata[0];
}

/* sum all entries in x and asign to y
 * block dim must be 256 */
__global__ void dot_stg_2_x(const FLOAT *x, FLOAT *y, int N)
{
    __shared__ FLOAT sdata[256];
    int idx = get_tid();
    int tid = threadIdx.x;
    int bid = get_bid();

    /* load data to shared mem */
    if (idx < N) {
        sdata[tid] = x[idx];
    }
    else {
        sdata[tid] = 0;
    }

    __syncthreads();

    /* reduction using shared mem */
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();

    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) y[bid] = sdata[0];
}

__global__ void dot_stg_3_x(FLOAT * partSum, FLOAT * y, FLOAT * x, FLOAT * u, int N, int row, int uInd)
{
    __shared__ FLOAT sdata[128];
    int tid = threadIdx.x;
    int i;

    sdata[tid] = 0;

    /* load data to shared mem */
    for (i = 0; i < N; i += 128) {
        if (tid + i < N) sdata[tid] += partSum[i + tid];
    }

    __syncthreads();

    /* reduction using shared mem */
    if (tid < 64) sdata[tid] = sdata[tid] + sdata[tid + 64];
    __syncthreads();

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) x[row] = (y[row] - sdata[0])/u[uInd];
}

/* d_part_sum_level_1 and d serve as cache: result stores in d[0] */
void cal_x_for_row(FLOAT *d_l_u, FLOAT *dy, FLOAT *d_part_sum_level_1, FLOAT * d_part_sum_level_2, FLOAT * dx, int N, int row)
{
    /* 1D block */
    int bs = 256;
    int len = N-row-1;
    /* 2D grid */
    int s = ceil(sqrt((len  + bs - 1.) / bs));
    dim3 grid = dim3(s, s);
    int gs = 0;
   
    

    /* stage 1 const FLOAT *u, FLOAT *x, FLOAT *z, int N, int row */
    dot_stg_1_x<<<grid, bs>>>(d_l_u, dx, d_part_sum_level_1, N, row);

    /* stage 2 */
    {
        /* 1D grid */
        int N2 = (len + bs - 1) / bs;

        int s2 = ceil(sqrt((N2 + bs - 1.) / bs));
        dim3 grid2 = dim3(s2, s2);

        dot_stg_2_x<<<grid2, bs>>>(d_part_sum_level_1, d_part_sum_level_2, N2);

        /* record gs */
        gs = (N2 + bs - 1.) / bs;
    }

    /* stage 3 FLOAT * partSum, FLOAT * y, FLOAT * x, FLOAT * u, int N, int row, int uInd)*/
    dot_stg_3_x<<<1, 128>>>(d_part_sum_level_2, dy, dx, d_l_u, gs, row, row*N + row);
}

#endif