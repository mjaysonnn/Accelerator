#include <stdio.h>
#include <stdlib.h>

__global__ void gpuMatMul(float * A, float * B, float *C,
                          int ROW_A, int COL_A, int COL_B) {
  /******************** TODO *********************/
  int j = blockIdx.x * blockDim.x + threadIdx.x;  //Block Thread의 Index에 Block Thread Size를 곱해서 Thread의 인덱스를 더한다.
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int k;
  float sum = 0.0f;

  if( i < ROW_A && j <COL_B){
    for (k=0; k<COL_A; k++){
      sum += A[i*COL_A +k] * B[k*COL_B + j];  //동시에 실행
    }
    C[i*COL_B +j] = sum;
  }
}

void mat_mul_cuda(float *A, float *B, float *C,
                    int ROW_A, int COL_A, int COL_B) {
  /******************** TODO *********************/
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, sizeof(float)*ROW_A*COL_A);
  cudaMalloc(&d_B, sizeof(float)*COL_A*COL_B);
  cudaMalloc(&d_C, sizeof(float)*ROW_A*COL_B);

  dim3 grid(COL_B/16 , ROW_A/16);
  dim3 block(16,16);

  cudaMemcpy(d_A, A, sizeof(float)*ROW_A*COL_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(float)*COL_A*COL_B, cudaMemcpyHostToDevice);

  gpuMatMul <<< grid, block >>> (d_A, d_B, d_C, ROW_A, COL_A, COL_B);

  cudaMemcpy(C, d_C, sizeof(float)*ROW_A*COL_B, cudaMemcpyDeviceToHost);  //blocking_true이다.

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
