#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <time.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include "reduction_aux.h"
#include "cal_y.h"
#include "cal_x.h"
#include <assert.h>
using namespace std;


void getIdentityMatrix(int n, FLOAT * array)
{
    int r = 0;
    int c = 0;

    for(r = 0; r < n; ++r)
    {
        for(c = 0; c < n; ++c)
        {
            if(r == c)
                array[r*n + c] = 1;
            else
                array[r*n + c] = 0;
        }
    }
}


void getInvertibleMatrix(int n, FLOAT* array)
{
    int i = 0;
    int j = 0;
    int k = 0;
    int mainRowNum = 0;
  
    FLOAT* tempArray = NULL;
  
    srand((int)time(NULL));
    int transformTime = (int)(rand()%1000);
    printf("We will do %d times tansformation.\n",transformTime);
 
    tempArray = (FLOAT*)malloc(sizeof(FLOAT)*n);
 
    for(i = 0; i < transformTime; ++i)
    {
        mainRowNum = (int)(rand()%(n-1));
        for(k = 0; k < n; ++k){
            int index = mainRowNum*n + k;
            if(((UINT16_MAX - (array[index])*((int)(rand()%5 - 10))) < 0) || ((UINT16_MAX*(-1)) - (array[index])*((int)(rand()%5 - 10)) > tempArray[k]))
                tempArray[k] = (array[index]);
            else
                tempArray[k] = (array[index])*((int)(rand()%5 - 10));
        }

        for(j = 0; j < n; ++j){
            if(mainRowNum != j)
                for(k = 0; k < n; ++k)
                {
                    int index = j*n +k;
                    if(((UINT16_MAX - array[index]) < tempArray[k]) || ((UINT16_MAX*(-1)) - array[index] > tempArray[k]))
                        array[index] = array[index]/4;
                    else
                        array[index] = array[index] + tempArray[k];
                }
        }
    }
 
    free(tempArray);
}


void initMatrix(FLOAT * a, int size){
  //FLOAT tempA[16] = {4,2,1,5,8,7,2,10,4,8,3,6,12,6,11,20};
  //FLOAT tempB[4] = {-2,-7,-7,-3};
  for(int i = 0; i < size; ++i){
      for(int j = 0; j < size; ++j){
          a[i * size + j] = 1.0 / (i + 1 + j + 1 - 1);//tempA[i * size + j];
          //l_u[i * size + j] = a[i * size + j];
      }
  }
}


void lud_base_optimized(FLOAT *a, int size)
{
    int i,j,k;
    FLOAT sum;
    for (i=0; i<size; i++)
    {
        //先计算左上角的U元素
        sum=a[i*size+i];
        for (k=0; k<i; k++) sum -= a[i*size+k]*a[k*size+i];
        a[i*size+i]=sum;
        //计算下侧的L矩阵部分
        for (j=i+1;j<size; j++)
        {
            sum=a[j*size+i];
            for (k=0; k<i; k++) sum -=a[j*size+k]*a[k*size+i];
            a[j*size+i]=sum/a[i*size+i];
        }
        //计算右侧的U矩阵部分
        for (j=i+1; j<size; j++)
        {
            sum=a[i*size+j];
            for (k=0; k<i; k++) sum -= a[i*size+k]*a[k*size+j];
            a[i*size+j]=sum;
        }
    }
}
void lud_base(int *a, int size)
{
     int i,j,k;
     int sum;
     for (i=0; i<size; i++)
     {
          //计算下侧的L矩阵部分
          for (j=i+1;j<size; j++)
          {
              sum=a[j*size+i];
              for (k=0; k<i; k++) sum -=a[j*size+k]*a[k*size+i];
              a[j*size+i]=sum;
          }
          //计算右侧的U矩阵部分
          for (j=i; j<size; j++)
          {
              sum=a[i*size+j];
              for (k=0; k<i; k++) sum -= a[i*size+k]*a[k*size+j];
              a[i*size+j]=sum;
          }
          //对下侧的L矩阵后续处理
          for (j=i+1;j<size; j++)
              a[j*size+i]=sum/a[i*size+i];
      }
}


void print_device_info(){
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
        (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }
    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;
    dev =0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf(" CUDA Driver Version / Runtime Version %d.%d / %d.%d\n",driverVersion/1000, (driverVersion%100)/10,runtimeVersion/1000, (runtimeVersion%100)/10);
    printf(" CUDA Capability Major/Minor version number: %d.%d\n",deviceProp.major, deviceProp.minor);
    printf(" Total amount of global memory: %.2f MBytes (%llu bytes)\n",(FLOAT)deviceProp.totalGlobalMem/(pow(1024.0,3)),(unsigned long long) deviceProp.totalGlobalMem);
    printf(" GPU Clock rate: %.0f MHz (%0.2f GHz)\n",deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
    printf(" Memory Clock rate: %.0f Mhz\n",deviceProp.memoryClockRate * 1e-3f);
    printf(" Memory Bus Width: %d-bit\n",deviceProp.memoryBusWidth);
    if (deviceProp.l2CacheSize) {
        printf(" L2 Cache Size: %d bytes\n",
        deviceProp.l2CacheSize);
    }

    printf(" Max Texture Dimension Size (x,y,z) 1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
    deviceProp.maxTexture1D , deviceProp.maxTexture2D[0],
    deviceProp.maxTexture2D[1],
    deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
    deviceProp.maxTexture3D[2]);

    printf(" Max Layered Texture Size (dim) x layers 1D=(%d) x %d, 2D=(%d,%d) x %d\n",
    deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
    deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
    deviceProp.maxTexture2DLayered[2]);

    printf(" Total amount of constant memory: %lu bytes\n",deviceProp.totalConstMem);
    printf(" Total amount of shared memory per block: %lu bytes\n",deviceProp.sharedMemPerBlock);
    printf(" Total number of registers available per block: %d\n",deviceProp.regsPerBlock);
    printf(" Warp size: %d\n", deviceProp.warpSize);
    printf(" Maximum number of threads per multiprocessor: %d\n",deviceProp.maxThreadsPerMultiProcessor);
    printf(" Maximum number of threads per block: %d\n",deviceProp.maxThreadsPerBlock);

    printf(" Maximum sizes of each dimension of a block: %d x %d x %d\n",
    deviceProp.maxThreadsDim[0],
    deviceProp.maxThreadsDim[1],
    deviceProp.maxThreadsDim[2]);

    printf(" Maximum sizes of each dimension of a grid: %d x %d x %d\n",
    deviceProp.maxGridSize[0],
    deviceProp.maxGridSize[1],
    deviceProp.maxGridSize[2]);

    printf(" Maximum memory pitch: %lu bytes\n", deviceProp.memPitch);
}


__global__ void divid(FLOAT * a, int head, int size){
  head = head -1;
  FLOAT headValue = a[(head)*size + head];
  //__syncthreads();
  int index = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if(index+head < size){
    a[(head+ index) * size + head] = a[(head + index) * size + head] / headValue;//headUValue[0];
    //printf("index %d (%d,%d) is: %f , headvalue: %f\n",index, head + index, head,  a[(head+index) * size + head], headValue);
  }
}


__global__ void updateSubCol(FLOAT * a, int head, int size){
  __shared__ FLOAT headLValue[1];
  head = head -1;
  if(threadIdx.x ==0 && threadIdx.y == 0){
    headLValue[0] = a[(blockIdx.y+1 +head)*size + head];
  }
  int step =  (size - head + 1)/blockDim.x;
  int idx = 0;
  int idy = 0;
  for(int i = 0; i <=step; i++){
    idx = threadIdx.x + i*blockDim.x + head + 1;
    idy = blockIdx.y+head+1;
    if(idx < size){
      a[idy * size + idx] = a[idy * size + idx] -  headLValue[0] * a[head * size + idx];
    }
  }
}


void lu_decomposition_gpu(int N, FLOAT * dev_a, int BLOCK_SIZE){
  int temp ;
  for(int k = 1 ;k < N;k++){
    temp = ceil(log(N-k)/log(2));
    BLOCK_SIZE = temp > 10 ? 1024:int(pow(2,temp));
    dim3 dimGrid(1,ceil((N-k)/(FLOAT)(BLOCK_SIZE)));
    dim3 dimBlock(1,BLOCK_SIZE);
    //printf("dimGrid: 1,%d; dimBlock: 1,%d \n", dimGrid.y, dimBlock.y);
    divid<< <dimGrid, dimBlock>> >(dev_a, k, N);
    dim3 dimGrid1(1,N-k);
    dim3 dimBlock1(BLOCK_SIZE,1);
    //printf("dimGrid1: 1,%d; dimBlock1: %d,1 \n", dimGrid1.y, dimBlock1.x);
    updateSubCol<< <dimGrid1, dimBlock1>> >(dev_a, k, N);
  }
  cudaDeviceSynchronize();
}


void lu_test(int dimension, bool isDebug){
    FLOAT *a,*result;
    int N = dimension;

    a = (FLOAT *)malloc(sizeof(FLOAT)*N*N);
    result = (FLOAT *)malloc(sizeof(FLOAT)*N*N);
    //initMatrix(a, N);
    getIdentityMatrix(N, a);
    getInvertibleMatrix(N, a);
    if(isDebug)
        print_matrix(a,N);
    
        FLOAT * dev_a;
    clock_t clock_start;
    clock_t clock_end;
    
    float elapsedTime = 0.0;
    cudaEvent_t event_start, event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);
    cudaEventRecord(event_start, 0);

    cudaMalloc ( (void**)&dev_a, N*N* sizeof (FLOAT) );
    cudaMemcpy(dev_a, a, N*N* sizeof (FLOAT), cudaMemcpyHostToDevice);
    int BLOCK_SIZE = 1024;
    lu_decomposition_gpu(N, dev_a, BLOCK_SIZE);
    cudaDeviceSynchronize();
    cudaMemcpy(result, dev_a, N*N* sizeof (FLOAT), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(event_stop, 0);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&elapsedTime, event_start, event_stop);
    cudaDeviceSynchronize();
  
    double clock_diff_sec ;
    printf("gpu time cost is %f s \n",elapsedTime/1000.0);
    if(isDebug)
        print_matrix(result,N);
  
    clock_start = clock();
    lud_base_optimized(a,N);
    clock_end = clock();
    clock_diff_sec = ((double)(clock_end - clock_start));
    printf("cpu time cost is %f s \n",clock_diff_sec/CLOCKS_PER_SEC);
    if(isDebug)
        print_matrix(a,N);

    checkDiff(a,result,N*N);

    cudaFree(dev_a);
    free(a);
    free(result);
}

void whole_test(int dimension, bool isDebug){
    cudaSetDevice(1);
    int N = dimension;
    int nbytes = N * N * sizeof(FLOAT);

    FLOAT *h_l_u = NULL, *h_y = NULL, *h_x = NULL, *h_b = NULL, *gpuS = NULL, *l_u_result =NULL;  //*sum = NULL, 
    FLOAT *d_l_u = NULL, *d_y = NULL, *d_x = NULL, *d_part_sum_level_1 = NULL, *d_part_sum_level_2 = NULL, *d_b =NULL;//, *dsum = NULL;
    int i, itr = 1;
    double td, th;

    /* allocate GPU mem */
    cudaMalloc((void **)&d_l_u, nbytes);
    cudaMalloc((void **)&d_y, N*sizeof(FLOAT));
    cudaMalloc((void **)&d_x, N*sizeof(FLOAT));
    cudaMalloc((void **)&d_b, N*sizeof(FLOAT));
    cudaMalloc((void **)&d_part_sum_level_1, sizeof(FLOAT) * ((N + 255) / 256));
    cudaMalloc((void **)&d_part_sum_level_2, sizeof(FLOAT) * ((N + 255) / 256));
    //cudaMalloc((void **)&dsum, sizeof(FLOAT) * N);

    if (d_l_u == NULL || d_y == NULL || d_part_sum_level_1 == NULL || d_part_sum_level_2 == NULL) {
        printf("couldn't allocate GPU memory\n");
        return;
    }

    printf("allocated %e MB on GPU\n", (nbytes +4*N*sizeof(FLOAT)  )/ (1024.f * 1024.f));

    /* alllocate CPU mem */
    h_l_u = (FLOAT *) malloc(nbytes);
    h_y = (FLOAT *) malloc(N*sizeof(FLOAT));
    h_x = (FLOAT *) malloc(N*sizeof(FLOAT));
    h_b = (FLOAT *) malloc(N*sizeof(FLOAT));
    l_u_result = (FLOAT *) malloc(nbytes);
    // sum = (FLOAT *) malloc(sizeof(FLOAT) * N);
    gpuS = (FLOAT *) malloc(sizeof(FLOAT) * N);

    if (h_l_u == NULL || h_y == NULL) {
        printf("couldn't allocate CPU memory\n");
        return;
    }
    printf("allocated %e MB on CPU\n", (nbytes +4* N*sizeof(FLOAT) ) / (1024.f * 1024.f));

    /* init */
    for (i = 0; i < N * N; i++) {
        if(i<N){
            h_b[i] = 1.0/(i%2+1);//1
            h_y[i] = h_b[i];
        }
    }
    // init a as a random invertibel matrix
    getIdentityMatrix(N, h_l_u);
    getInvertibleMatrix(N, h_l_u);

    /* copy data to GPU */
    cudaMemcpy(d_l_u, h_l_u, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(FLOAT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(FLOAT), cudaMemcpyHostToDevice);
    //cudaMemcpy(dsum, sum, N*sizeof(FLOAT), cudaMemcpyHostToDevice);

    /* let dust fall */
    cudaThreadSynchronize();
    td = get_time();

    lu_decomposition_gpu(N, d_l_u, 1024);
    cudaMemcpy(l_u_result, d_l_u, N*N* sizeof (FLOAT), cudaMemcpyDeviceToHost);

    /* call GPU */
    for (int j = 0; j < itr; j++){
        for (i = 1; i < N; i++) {
            dot_device(d_l_u, d_y, d_part_sum_level_1, d_part_sum_level_2, d_b, N, i);
            //dot<<<N, TileSize>>>(d_l_u, d_y, dsum, i, N);
        }
        //cudaThreadSynchronize();
        cudaMemcpy(d_x, d_y, N*sizeof(FLOAT), cudaMemcpyDeviceToDevice);
        //cudaThreadSynchronize();
        for (i = N-2; i >= 0; i--) {
            cal_x_for_row(d_l_u, d_y, d_part_sum_level_1, d_part_sum_level_2, d_x, N, i);
            //cudaThreadSynchronize();
            //dot<<<N, TileSize>>>(d_l_u, d_y, dsum, i, N);
        }
    }
            
    // cudaMemcpy(gpuS, d_y, N*sizeof(FLOAT), cudaMemcpyDeviceToHost);
    // printf("gpu d_y \n");
    // print_vec(gpuS,N);
    cudaMemcpy(gpuS, d_x, N*sizeof(FLOAT), cudaMemcpyDeviceToHost);
    /* let GPU finish */
    cudaThreadSynchronize();
    td = get_time() - td;

    th = get_time();
    lud_base_optimized(h_l_u,N);
    h_y[0] = h_b[0];
    for(int i=1; i<N; i++)
    {
        FLOAT sum=0;
        for(int j = 0; j < i; j++){
            sum+=h_l_u[i*N +j]*h_y[j];
        }
        h_y[i]=(h_b[i]-sum);
    }
    h_x[N-1]=h_y[N-1];///h_l_u[N*N-1];//求解X[N-1]
    for(int i = N-2; i >= 0; i--){
        FLOAT sum = 0.0;
        for(int j = N-1; j >= i+1; j--){
            sum += h_l_u[i*N + j] * h_x[j];
        }
        h_x[i] = (h_y[i] - sum)/h_l_u[i*N + i];
    }
    //printf("\n");
    th = get_time() - th;

    /* copy data from GPU */
    //cudaMemcpy(&asd, d, sizeof(FLOAT), cudaMemcpyDeviceToHost);

    //printf("dot, answer: %d, calculated by GPU:%f, calculated by CPU:%f\n", 2 * N, asd, ash);
    checkDiff(h_l_u,l_u_result,N*N);
    FLOAT diff = checkDiff(h_x,gpuS,N);
    //printf("y diff is %.10f \n",diff);
    printf("x diff is %.10f \n",diff);
    printf("GPU time: %e, CPU time: %e, speedup: %g\n", td, th, th / td);

    if(isDebug){
        printf("h_l_u \n");
        print_matrix(h_l_u, N);
        // printf("y \n");
        // print_matrix(h_y, N);
        printf("h_b \n");
        print_vec(h_b,N);
        printf("cpu h_y \n");
        print_vec(h_y,N);
        printf("cpu h_x \n");
        print_vec(h_x,N);
        printf("gpu \n");
        print_vec(gpuS,N);
    }
    
    cudaFree(d_l_u);
    cudaFree(d_y);
    cudaFree(d_part_sum_level_1);
    cudaFree(d_part_sum_level_2);
    cudaFree(d_b);
    cudaFree(d_x);

    free(h_l_u);
    free(h_y);
    free(h_x);
    free(h_b);
    free(gpuS);
}


int main(int argc, char *argv[]){
    int dimension = 18;
    bool isDebug = false;
    for(int i=1;i<argc;i++)
    {
      cout<<"argument["<<i<<"] is: "<<argv[i]<<endl;
    }
    // dimension = stoi(argv[1], 0, 10);
    // if(argv[2][0] == 't')
    //     isDebug = true;
    //lu_test(dimension, isDebug)
    whole_test(dimension, isDebug);
    return 0;

}