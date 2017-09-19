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

  float *d_A, *d_B, *d_C;
  float *d_trC;
  float one = 1;
  float zero = 0;

  int M = ROW_A;
  int K = COL_A;
  int N = COL_B;


  //if(ROW_A != COL_A || ROW_A != COL_B)
  //{
  //  printf("Support Square Matrix Only!\n");
  // exit(EXIT_FAILURE);
  //}
  /******************** TODO *********************/

  cublasHandle_t handle;
  cublasStatus_t status;

  status = cublasCreate(&handle);
  CHECK_CUBLAS_ERROR(status);

  cudaMalloc(&d_A, sizeof(float)*M*K);
  cudaMalloc(&d_B, sizeof(float)*K*N);
  cudaMalloc(&d_C, sizeof(float)*M*N);
  cudaMalloc(&d_trC, sizeof(float)*N*M);


  status = cublasSetMatrix(M, K, sizeof(float), A, M, d_A, M);
  CHECK_CUBLAS_ERROR(status);

  status = cublasSetMatrix(K, N, sizeof(float), B, K, d_B, K);
  CHECK_CUBLAS_ERROR(status);

  status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &one, d_A, K, d_B, N, &zero, d_trC, M);  // Non_Blocking 이다.
  CHECK_CUBLAS_ERROR(status);

  status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, &one, d_trC, N, &zero, d_trC, M, d_C, M);
  CHECK_CUBLAS_ERROR(status);

  status = cublasGetMatrix(M, N, sizeof(float), d_C, M, C, M);
  CHECK_CUBLAS_ERROR(status);

  status = cublasDestroy(handle);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
