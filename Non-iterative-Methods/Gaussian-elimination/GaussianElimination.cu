#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

// includes, kernels
#include "GaussianEliminationKernal.cu"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

extern "C" int compute_gold(float*, const float*, unsigned int);
Matrix allocate_matrix_on_gpu(const Matrix M);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost);
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice);
void gauss_eliminate_on_device(const Matrix M, Matrix P);
void gauss_eliminate_on_device_optimized(const Matrix M, Matrix P);
int perform_simple_check(const Matrix M);
void print_matrix(const Matrix M);
void write_matrix_to_file(const Matrix M);
float get_random_number(int, int);
void checkCUDAError(const char *msg);
int checkResults(float *U, float *gpu_result, int num_elements, float threshold);


int main(int argc, char** argv) 
{
    // Matrices for the program
	Matrix  A; // The NxN input matrix
	Matrix  U; // The upper triangular matrix 
	struct timeval start, stop;
	// Initialize the random number generator with a seed value 
	srand(time(NULL));
	
	// Check command line arguments
	if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}		
	 
	// Allocate and initialize the matrices
	A  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1);
	U  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); 

	// Perform Gaussian elimination on the CPU 
	Matrix reference = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	gettimeofday(&start, NULL);
	int status = compute_gold(U.elements, A.elements, A.num_rows);
	gettimeofday(&stop, NULL);
	printf("Execution time gold = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
	if(status == 0){
		printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
		exit(0);
	}
	status = perform_simple_check(U); // Check that the principal diagonal elements are 1 
	if(status == 0){
		printf("The upper triangular matrix is incorrect. Exiting. \n");
		exit(0); 
	}
	printf("Gaussian elimination on the CPU was successful. \n");

	// Perform the vector-matrix multiplication on the GPU. Return the result in U
	gauss_eliminate_on_device(A,U);
	int num_elements = MATRIX_SIZE*MATRIX_SIZE;

	int res = checkResults(U.elements, U.elements, num_elements, 0.001f);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	gauss_eliminate_on_device_optimized(A, U);
    // check if the device result is equivalent to the expected solution
	res = checkResults(U.elements, U.elements, num_elements, 0.001f);
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	// Free host matrices
	free(A.elements); A.elements = NULL;
	free(U.elements); U.elements = NULL;
	free(U.elements); U.elements = NULL;

	return 0;
}

void gauss_eliminate_on_device(const Matrix A, Matrix U)
{
	struct timeval start,stop;

	Matrix gpu_u = allocate_matrix_on_gpu( U );

	//Copy matrices to gpu, copy A right into U
	copy_matrix_to_device( gpu_u, A );
	
	int num_blocks = 1;
	
	int threads_per_block = 512;
	
	int ops_per_thread = MATRIX_SIZE / (threads_per_block*num_blocks);
	
	printf("== GPU (Slow) ==\n");
	printf("	Threads per block: %d\n",threads_per_block);
	printf("	Number of blocks: %d\n",num_blocks);
	printf("	Operations per thread: %d\n",ops_per_thread);
	
	dim3 thread_block(threads_per_block, 1, 1);
	dim3 grid(num_blocks,1);
	
	gettimeofday(&start, NULL);
	
	// Launch the kernel <<<grid, thread_block>>>
	gauss_eliminate_kernel<<<grid, thread_block>>>(gpu_u.elements,ops_per_thread);
	
	//Sync at end and check for errors
	cudaThreadSynchronize();
	checkCUDAError("FAST KERNEL FAILURE");
	gettimeofday(&stop, NULL);
	printf("Execution time GPU = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

	//Copy data back
	copy_matrix_from_device(U, gpu_u);
	
	//Free memory on device
	cudaFree(gpu_u.elements);
	}

void gauss_eliminate_on_device_optimized(const Matrix A, Matrix U){
    printf("== GPU (Fast) ==\n");
	Matrix gpu_u = allocate_matrix_on_gpu( U );

	//Copy matrices to gpu, copy A right into U
	copy_matrix_to_device( gpu_u, A );

	//Each thread within a block will take some j iterations
	int threads_per_block = 256; 
	struct timeval start,stop;
	int stride = threads_per_block;
	printf("	Threads per block / stride: %d\n",threads_per_block);

	
	
	int k;
	gettimeofday(&start, NULL);

	for(k = 0; k < MATRIX_SIZE; k++)
	{
		int isize = (MATRIX_SIZE-1) - (k+1) + 1;
		int num_blocks = isize;
		if(num_blocks <= 0)
		{
			num_blocks = 1;
		}
		
		dim3 thread_block(threads_per_block, 1, 1);
		dim3 grid(num_blocks,1);
	
		gauss_eliminate_kernel_optimized_div<<<grid, thread_block>>>(
			gpu_u.elements,
			k,
			stride);
		gauss_eliminate_kernel_optimized<<<grid, thread_block>>>(
			gpu_u.elements,
			k,
			stride);
			cudaThreadSynchronize();
			
		checkCUDAError("FAST KERNEL FAILURE");
	}
	gettimeofday(&stop, NULL);
	printf("Execution time GPU = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

	
	//Sync at end
	cudaThreadSynchronize();

	//Copy data back
	copy_matrix_from_device(U, gpu_u);
	
	//Free memory on device
	cudaFree(gpu_u.elements);
	
	
	int i, j;
	for(i = 0; i < MATRIX_SIZE; i++)
        for(j = 0; j < i; j++) 
        {
            U.elements[i * MATRIX_SIZE + j] = 0.0;
        }
}

// Allocate a device matrix of same size as M.
Matrix allocate_matrix_on_gpu(const Matrix M){
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix allocate_matrix(int num_rows, int num_columns, int init){
    	Matrix M;
    	M.num_columns = M.pitch = num_columns;
    	M.num_rows = num_rows;
    	int size = M.num_rows * M.num_columns;
		
	M.elements = (float*) malloc(size*sizeof(float));
	for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void  copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice){
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Prints the matrix out to screen
void print_matrix(const Matrix M){
	for(unsigned int i = 0; i < M.num_rows; i++){
		for(unsigned int j = 0; j < M.num_columns; j++)
			printf("%f ", M.elements[i*M.num_rows + j]);
		printf("\n");
	} 
	printf("\n");
}

// Returns a random floating-point number between the specified min and max values 
float get_random_number(int min, int max){
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}

// Performs a simple check on the upper triangular matrix. Checks to see if the principal diagonal elements are 1
int perform_simple_check(const Matrix M){
	for(unsigned int i = 0; i < M.num_rows; i++)
        if((fabs(M.elements[M.num_rows*i + i] - 1.0)) > 0.001) return 0;
	
    return 1;
} 

// Writes the matrix to a file 
void write_matrix_to_file(const Matrix M){
	FILE *fp;
	fp = fopen("matrix.txt", "wt");
	for(unsigned int i = 0; i < M.num_rows; i++){
        for(unsigned int j = 0; j < M.num_columns; j++)
            fprintf(fp, "%f", M.elements[i*M.num_rows + j]);
        }
    fclose(fp);
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) 
	{
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}						 
}

int checkResults(float *U, float *gpu_result, int num_elements, float threshold)
{
    int checkMark = 1;
    float epsilon = 0.0;
    
    for(int i = 0; i < num_elements; i++)
        if(fabsf((U[i] - gpu_result[i])/U[i]) > threshold){
            checkMark = 0;
            break;
        }

    for(int i = 0; i < num_elements; i++)
        if(fabsf((U[i] - gpu_result[i])/U[i]) > epsilon){
            epsilon = fabsf((U[i] - gpu_result[i])/U[i]);
        }

    printf("Max epsilon = %f. \n", epsilon); 
    return checkMark;
}
