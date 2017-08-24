#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

#define CHECK_CUBLAS_ERROR(err) \
if (err != CUBLAS_STATUS_SUCCESS) \
    {\
        printf("[%s:%d] CUBLAS error %d\n", __FILE__, __LINE__, err);\
        exit(EXIT_FAILURE);\
    } 

void mat_mul_cublas(float *A, float *B, float *C,
                    int ROW_A, int COL_A, int COL_B) {
  
  float *d_A, *d_B, *d_C, *d_trC;
  float one = 1;
  float zero = 0;
  
  int N = ROW_A;
  
  

  if(ROW_A != COL_A || ROW_A != COL_B)
  {
    printf("Support Square Matrix Only!\n");
   exit(EXIT_FAILURE);
  }
  /******************** TODO *********************/
  
  cublasHandle_t handle;
  cublasStatus_t status;

  status = cublasCreate(&handle);
  CHECK_CUBLAS_ERROR(status);

  cudaMalloc(&d_A, sizeof(float)*N*N);
  cudaMalloc(&d_B, sizeof(float)*N*N);
  cudaMalloc(&d_C, sizeof(float)*N*N);
  cudaMalloc(&d_trC, sizeof(float)*N*N);


  status = cublasSetMatrix(N, N, sizeof(float), A, N, d_A, N);
  CHECK_CUBLAS_ERROR(status);

  status = cublasSetMatrix(N, N, sizeof(float), B, N, d_B, N);
  CHECK_CUBLAS_ERROR(status);

  status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &one, d_A, N, d_B, N, &zero, d_trC, N);  // Non_Blocking 이다.
  CHECK_CUBLAS_ERROR(status);

  status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, &one, d_trC, N, &zero, d_trC, N, d_C, N);
  CHECK_CUBLAS_ERROR(status);

  status = cublasGetMatrix(N, N, sizeof(float), d_C, N, C, N);
  CHECK_CUBLAS_ERROR(status);

  status = cublasDestroy(handle);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

