#include <stdio.h>
#include <stdlib.h>

extern int N;

__global__ void gpuReduction(int *g_num,
                             int *g_sum,
                             int TotalNum) {

  __shared__ int l_sum[block_dim];
  int i =blockIdx.x * blockDim.x + threadIdx.x;
  int l_i = threadIdx.x;

  l_sum[l_i]=(i < TotalNum) ? g_num[i] : 0;
  __syncthreads();

  for(int p = blockDim.x/2 ; p >= 1; p = p>>1){
    if (l_i < p) l_sum[l_i] += l_sum[l_i + p];
    __syncthreads();
  }

  if(l_i == 0){
    g_sum[blockIdx.x] = l_sum[0];
  }

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
