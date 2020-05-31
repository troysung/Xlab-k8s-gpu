
#ifndef REDUCTION_AUX
#define REDUCTION_AUX

#include <stdio.h>
#include <cuda.h>

typedef double FLOAT;
#define USE_UNIX 1

/* get thread id: 1D block and 2D grid */
#define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)

/* get block id: 2D grid */
#define get_bid() (blockIdx.x + blockIdx.y * gridDim.x)

/* warm up, start GPU, optional */
void warmup();

/* get time stamp */
double get_time(void);

/* asum host */
FLOAT asum_host(FLOAT *x, int N);

/* a little system programming */
#if USE_UNIX

#include <sys/time.h>
#include <time.h>

double get_time(void)
{
    struct timeval tv;
    double t;

    gettimeofday(&tv, (struct timezone *)0);
    t = tv.tv_sec + (double)tv.tv_usec * 1e-6;

    return t;
}
#else
#include <windows.h>

double get_time(void)
{
    LARGE_INTEGER timer;
    static LARGE_INTEGER fre;
    static int init = 0;
    double t;

    if (init != 1) {
        QueryPerformanceFrequency(&fre);
        init = 1;
    }

    QueryPerformanceCounter(&timer);

    t = timer.QuadPart * 1. / fre.QuadPart;

    return t;
}
#endif

/* warm up GPU */
__global__ void warmup_knl()
{
    int i, j;

    i = 1;
    j = 2;
    i = i + j;
}
__device__ void warpReduce(volatile FLOAT *sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

void warmup()
{
    int i;

    for (i = 0; i < 8; i++) {
        warmup_knl<<<1, 256>>>();
    }
}

/* host, add */
FLOAT asum_host(FLOAT *x, int N)
{
    int i;
    FLOAT t = 0;

    for (i = 0; i < N; i++) t += x[i];

    return t;
}
float checkDiff(FLOAT * a, FLOAT * b,int N){
    float sum = 0;
    float max = 0;
    int pos;
    for(int i = 0; i< N; i++){
           sum += abs(a[i] - b[i]);
           if(abs(a[i] - b[i]) > max){
                max = abs(a[i] - b[i]);
                pos = i;
           }
    }
    int n = sqrt(N);
    printf("max diff is %.10f avg diff is %.10f ,pos (%d,%d), sum %.10f\n",max, sum/(float)N, pos/n, pos%n, sum);
    return sum;
}
void print_matrix(FLOAT * matrix, int size){
  for (int i=0; i<size; i++)
  {
      for(int j=0; j<size; j++){
          printf("%f  ",matrix[i*size + j ]);
      }
      printf("\n");
  }
}
void print_vec(FLOAT * vec, int size){
  for (int i=0; i<size; i++)
  {
      printf("%f  ",vec[i]);
  }
  printf("\n");
}

#endif
