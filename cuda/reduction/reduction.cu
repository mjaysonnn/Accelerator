#include <stdio.h>
#include <stdlib.h>

extern int N;

__global__ void gpuReduction(int *g_num,
                             int *g_sum,
                             int TotalNum) {

  int i =blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0.0f;
  
  // TODO: implement kernel code here
  // int j = blockIdx.x * blockDim.x + threadIdx.x;
  // int i = blockIdx.y * blockDim.y + threadIdx.y;
  // int k;
  // float sum = 0.0f;
  //
  // if( i < ROW_A && j <COL_B){
  //   for (k=0; k<COL_A; k++){
  //     sum += A[i*COL_A +k] * B[k*COLB + j];
  //   }
  //   C[i*COL_B +j] = sum;
  // }
  // __kernel void reduction(__global int *g_num,
  //                         __global int *g_sum,
  //                         __local int *l_sum,
  //                         int TotalNum) {
  //   int i = get_global_id(0);
  //   int l_i = get_local_id(0);
  //
  //   l_sum[l_i] = (i < TotalNum) ? g_num[i] : 0;  // 이건 뭔 개소리일까
  //   barrier(CLK_LOCAL_MEM_FENCE);
  //
  //   for (int p = get_local_size(0) / 2; p >= 1; p = p >> 1) {
  //     if (l_i < p) l_sum[l_i] += l_sum[l_i + p];
  //     barrier(CLK_LOCAL_MEM_FENCE);
  //   }
  //
  //   if (l_i == 0) {
  //     g_sum[get_group_id(0)] = l_sum[0];
  //   }
  // }

}

double reduction_cuda(int *array, int N) {
  // TODO: implement host code here
  int sum=0;
  int i;
  size_t block_dim =256;
  size_t num_blocks = N / block_dim;

  int *d_array;
  int *d_partial_sum;

  int *partial_sum=(int *)malloc(sizeof(int)*num_blocks);

  cudaMalloc(&d_array, sizeof(int)*N);
  cudaMalloc(&d_partial_sum, sizeof(int)*num_blocks);

  cudaMemcpy(d_array, array, sizeof(int)*N, cudaMemcpyHostToDevice);

  dim3 grid(num_blocks);
  dim3 block(block_dim);

  gpuReduction <<< grid, block , sizeof(int)*block_dim >>>(d_array, d_partial_sum, N);

  cudaMemcpy(partial_sum, d_partial_sum, sizeof(int)*num_blocks, cudaMemcpyDeviceToHost);

  for( i=0; i<num_blocks; i++){
    sum += partial_sum[i];
  }
  free(partial_sum);
  free(d_array);
  cudaFree(d_partial_sum);
  return (double)sum / N ;


}
