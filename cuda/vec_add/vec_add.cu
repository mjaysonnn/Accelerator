#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
__global__ void gpuVecAdd(float *A, float *B, float *C) {   //device용이라는 __global__ 로 알수있다.
  // TODO: write kernel code here
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  C[tid]=A[tid]+B[tid];
}

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

void init(float *V, int N) {
  for (int i = 0; i < N; i++) {
    V[i] = rand() % 100;
  }
}

void verify(float *A, float *B, float *C, int N) {
  for (int i = 0; i < 16384; i++) {
    if (A[i] + B[i] != C[i]) {
      printf("Verification failed! A[%d] = %d, B[%d] = %d, C[%d] = %d\n",
             i, A[i], i, B[i], i, C[i]);
      return;
    }
  }
  printf("Verification success!\n");
}

int main() {
  int N = 16384;

  float *A = (float*)malloc(sizeof(float) * N);
  float *B = (float*)malloc(sizeof(float) * N);
  float *C = (float*)malloc(sizeof(float) * N);

  init(A, N);
  init(B, N);

  // Memory objects of the device
  float *d_A, *d_B, *d_C;



  // TODO: allocate memory objects d_A, d_B, and d_C.
  cudaMalloc(&d_A, sizeof(float)*N);
  cudaMalloc(&d_C, sizeof(float)*N);
  cudaMalloc(&d_B, sizeof(float)*N);


  // TODO: copy "A" to "d_A" (host to device).
  cudaMemcpy(d_A, A, sizeof(float)*N, cudaMemcpyHostToDevice);

  // TODO: copy "B" to "d_B" (host to device).
  cudaMemcpy(d_B, B, sizeof(float)*N, cudaMemcpyHostToDevice);
  // TODO: launch the kernel.
  dim3 dimBlock(32,1); // 스레드 블록의 크기를 지정
  dim3 dimGrid(N/32,1); //grid의 크기를 지정 , global_size랑 다름 총 몇개의 thread_block이 있냐

  double start_time = get_time();

  gpuVecAdd <<< dimGrid, dimBlock >>> (d_A, d_B, d_C);  //Background에서 돈다
  // TODO: copy "d_C" to "C" (device to host).
  cudaMemcpy(C, d_C, sizeof(float)*N, cudaMemcpyDeviceToHost); // MemCopy 끝난게 Kernel이 끝나다는 뜻이다.
  double end_time = get_time();
  printf("Elapsed time: %f sec\n", end_time - start_time);
  verify(A, B, C, N);

  // TODO: release d_A, d_B, and d_C.
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(C);

  return 0;
}

/*
Elapsed time: 0.000117 sec
Verification success!
*/
