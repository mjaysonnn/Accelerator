#include <stdio.h>
#include <stdlib.h>

#include <stdio.h>
#include <stdlib.h>

__global__ void gpuMatMul(float * A, float * B, float *C,
                          int ROW_A, int COL_A, int COL_B);
void mat_mul_cuda_multi(float *A, float *B, float *C,
                  int ROW_A, int COL_A, int COL_B) {
  /******************** TODO *********************/
  float *d_A[4], *d_B[4], *d_C[4];
  for (int device =0; device < 4; device++){
    cudaSetDevice(device);   // 현재 Context 에서 사용할 CUDA device를 선택
    cudaMalloc(&d_A[device], sizeof(float)*(ROW_A/4)*COL_A); //cudaMalloc시 현재 사용중인 device에 메모리가 할당됨
    cudaMalloc(&d_B[device], sizeof(float)*(COL_A)*COL_B);
    cudaMalloc(&d_C[device], sizeof(float)*(ROW_A/4)*COL_B);
    cudaMemcpy(d_A[device], A+(ROW_A/4)*COL_A*device, sizeof(float)*(ROW_A/4)*COL_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B[device], B, sizeof(float)*(COL_A)*COL_B, cudaMemcpyHostToDevice);

  }
  dim3 grid(COL_B/16, (ROW_A/4)/16);
  dim3 block(16,16);

  for(int device = 0; device <4; device++){  //커널 실행될때 cudaSetDevice 사용
    cudaSetDevice(device);
    gpuMatMul <<< grid, block >>>(d_A[device],d_B[device],d_C[device], ROW_A, COL_A, COL_B);
  }
  for(int device =0 ; device <4; device++){
    cudaSetDevice(device);
    cudaMemcpy(C+(ROW_A/4)*COL_B*device, d_C[device], sizeof(float)*(ROW_A/4)*COL_B, cudaMemcpyDeviceToHost);
  }
  for (int device=0; device <4; device++){
    cudaFree(d_A[device]);
    cudaFree(d_B[device]);
    cudaFree(d_C[device]);  // 메모리 해제
  }
}
