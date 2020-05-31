#ifndef CAL_Y
#define CAL_Y

#include "reduction_aux.h"
#include <assert.h>
/* host, add */
void dot_host(FLOAT *x, FLOAT *y, int N, FLOAT * sum)
{
    int i;
    FLOAT t = 0;

    assert(x != NULL);
    assert(y != NULL);
    for (i = 1; i < N; i++){
        t = 0;
        for(int j = 0; j < i; j++)
            t += x[i*N + j] * y[i*N+j];
        sum[i] = t;
    }
}


/* partial dot product */
__global__ void dot_stg_1(const FLOAT *l, FLOAT *y, FLOAT *z, int N, int row)
{
    __shared__ FLOAT sdata[256];
    int endPos = row * N + row;
    int idx = get_tid() + row * N;
    int yInd = get_tid();
    int tid = threadIdx.x;
    int bid = get_bid();

    /* load data to shared mem */
    if (idx < endPos) {
        sdata[tid] = l[idx] * y[yInd];
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
__global__ void dot_stg_2(const FLOAT *x, FLOAT *y, int N)
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

__global__ void dot_stg_3(FLOAT *x, int N, FLOAT * y, int row, FLOAT * b)
{
    __shared__ FLOAT sdata[128];
    int tid = threadIdx.x;
    int i;

    sdata[tid] = 0;

    /* load data to shared mem */
    for (i = 0; i < N; i += 128) {
        if (tid + i < N) sdata[tid] += x[i + tid];
    }

    __syncthreads();

    /* reduction using shared mem */
    if (tid < 64) sdata[tid] = sdata[tid] + sdata[tid + 64];
    __syncthreads();

    if (tid < 32) warpReduce(sdata, tid);

    if (tid == 0) y[row] = b[row] - sdata[0];
}

/* dz and d serve as cache: result stores in d[0] */
void dot_device(FLOAT *dx, FLOAT *dy, FLOAT *dz, FLOAT *d, FLOAT * b, int N, int row)
{
    /* 1D block */
    int bs = 256;

    /* 2D grid */
    int s = ceil(sqrt((row  + bs - 1.) / bs));
    dim3 grid = dim3(s, s);
    int gs = 0;
   
    /* stage 1 */
    dot_stg_1<<<grid, bs>>>(dx, dy, dz, N, row);

    /* stage 2 */
    {
        /* 1D grid */
        int N2 = (row  + bs - 1) / bs;

        int s2 = ceil(sqrt((N2 + bs - 1.) / bs));
        dim3 grid2 = dim3(s2, s2);

        dot_stg_2<<<grid2, bs>>>(dz, d, N2);

        /* record gs */
        gs = (N2 + bs - 1.) / bs;
    }

    /* stage 3 */
    dot_stg_3<<<1, 128>>>(d, gs, dy, row, b);
}

#endif