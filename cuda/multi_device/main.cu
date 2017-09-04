#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(int * mem, int did){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int tsz = blockDim.x * gridDim.x;
  mem[tid] = did * tsz + tid;
}

int main(){
  int deviceCount;
  int **deviceMem;
  int *result;
  int threadPerDevice =3;
  cudaGetDeviceCount(&deviceCount);
  printf("%d devices available\n", deviceCount);
  deviceMem = (int **)malloc(sizeof(int)*deviceCount);
  result = (int*)malloc(sizeof(int)*deviceCount*threadPerDevice);
  for(int device =0; device < deviceCount; ++device){
    cudaSetDevice(device);
    cudaMalloc(&deviceMem[device], sizeof(int)*threadPerDevice);
  }
  for(int device =0; device < deviceCount; ++device){
    cudaSetDevice(device);
    kernel <<< 1, threadPerDevice >>> (deviceMem[device], device);
  }
  for(int device =0; device < deviceCount; ++device){
    cudaSetDevice(device);
    cudaMemcpy(result+device*threadPerDevice, deviceMem[device],
      sizeof(int)*threadPerDevice, cudaMemcpyDeviceToHost);
  }
  for(int i=0 ; i < threadPerDevice*deviceCount; i++){
    printf("%d\n", result[i]);
  }
  return 0;
}
