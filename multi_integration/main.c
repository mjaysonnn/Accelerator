#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <unistd.h>

static int N = 16777216;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

double reduction_seq(int *array, int N);
double reduction_opencl(int *array, int N);

int main() {
  int *array = (int*)malloc(sizeof(int) * N);
  int i;
  double ans_seq, ans_opencl;

  for (i = 0; i < N; i++) {
    array[i] = rand() % 100;
  }

  printf("Sequential version...\n");
  ans_seq = reduction_seq(array, N);
  printf("Average: %f\n", ans_seq);

  printf("OpenCL version...\n");
  ans_opencl = reduction_opencl(array, N);
  printf("Average: %f\n", ans_opencl);

  free(array);
  return 0;
}

double reduction_seq(int *array, int N) {
  int sum = 0;
  int i;
  double start_time, end_time;
  start_time = get_time();
  for (i = 0; i < N; i++) {
    sum += array[i];
  }
  end_time = get_time();
  printf("Elapsed time: %f sec\n", end_time - start_time);
  return (double)sum / N;
}

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

char *get_source_code(const char *file_name, size_t *len) {
  char *source_code;
  size_t length;
  FILE *file = fopen(file_name, "r");
  if (file == NULL) {
    printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
    exit(EXIT_FAILURE);
  }

  fseek(file, 0, SEEK_END);
  length = (size_t)ftell(file);
  rewind(file);

  source_code = (char *)malloc(length + 1);
  fread(source_code, length, 1, file);
  source_code[length] = '\0';

  fclose(file);

  *len = length;
  return source_code;
}

double reduction_opencl(int *array, int N) {
  cl_platform_id platform;

  cl_device_id *device;

  cl_context context;

  cl_command_queue *queue;

  cl_program program;

  char *kernel_source;

  size_t kernel_source_size;

  cl_kernel *kernel;

  cl_int err;

  int i;

  double start_time, end_time;

  err = clGetPlatformIDs(1, &platform, NULL); CHECK_ERROR(err);

  // err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  // CHECK_ERROR(err);

  cl_uint num_devices;

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices); CHECK_ERROR(err);

  printf("%u devices\n", num_devices);

  device = (cl_device_id*)malloc(sizeof(cl_device_id)*num_devices);
  kernel = (cl_kernel*)malloc(sizeof(cl_kernel)*num_devices);
  queue = (cl_command_queue*)malloc(sizeof(cl_command_queue)*num_devices);

  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, device, NULL); CHECK_ERROR(err);

  context = clCreateContext(NULL, num_devices, device, NULL, NULL, &err);  CHECK_ERROR(err);

  for (i=0; i<num_devices; i++){
  queue[i] = clCreateCommandQueue(context, device[i], 0, &err); CHECK_ERROR(err);
  }

  kernel_source = get_source_code("kernel.cl", &kernel_source_size);

  program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, &kernel_source_size, &err); CHECK_ERROR(err);

  err = clBuildProgram(program, num_devices, device, "", NULL, NULL);  // 4개의 device에서 프로그램을 빌드한다.
  if (err == CL_BUILD_PROGRAM_FAILURE) {
      size_t log_size;
      char *log;

      err = clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);  // 그리고 오류를 체크하기 위해서 여러 Device중에 첫번째꺼를 고른다는 뜻이다.
      CHECK_ERROR(err);

      log = (char*)malloc(log_size+1);

      err = clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);  CHECK_ERROR(err);

      log[log_size] = '\0';
      printf("Compiler error:\n%s\n", log);
      free(log);
      exit(0);
  }
  CHECK_ERROR(err);

  for(i=0; i<num_devices; i++){
  kernel[i]= clCreateKernel(program, "integral", &err);  CHECK_ERROR(err);
  }
  start_time = get_time();
  size_t global_size = N/num_devices;
  size_t local_size = 256;
  size_t num_work_groups = global_size / local_size;

  cl_mem  *buf_partial_sum;
  double *partial_sum;

  buf_partial_sum = (cl_mem*)malloc(sizeof(cl_mem)*num_devices);
  partial_sum = (double*)malloc(sizeof(double)*num_devices*num_work_groups);
  for(i=0; i<num_devices; i++){

  buf_partial_sum[i]=clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*num_work_groups, NULL, &err); CHECK_ERROR(err);
  }

  int base, offset;

  for(i=0; i<num_devices; i++){
  base = N / num_devices * i;
  offset = N/num_devices;
  err = clSetKernelArg(kernel[i], 0, sizeof(int), &N); CHECK_ERROR(err);
  err = clSetKernelArg(kernel[i], 1, sizeof(int), &base); CHECK_ERROR(err);
  err = clSetKernelArg(kernel[i], 2, sizeof(int), &offset); CHECK_ERROR(err);
  err = clSetKernelArg(kernel[i], 3, sizeof(cl_mem), &buf_partial_sum[i]);  CHECK_ERROR(err);
  err = clSetKernelArg(kernel[i], 4, sizeof(double)*local_size, NULL);   CHECK_ERROR(err);
  }

  for(i=0; i<num_devices; i++){
  err = clEnqueueNDRangeKernel(queue[i], kernel[i], 1, NULL, &global_size, &local_size, 0, NULL, NULL); CHECK_ERROR(err);
  }
  for(i=0; i<num_devices; i++){
    err = clEnqueueReadBuffer(queue[i], buf_partial_sum[i], CL_FALSE, 0, sizeof(double)*num_work_groups, partial_sum+num_devices*i, 0, NULL, NULL);  CHECK_ERROR(err);
  }
  for(i=0; i<num_devices; i++){
    err = clFinish(queue[i]); CHECK_ERROR(err);
  }
  double sum = 0;
  for (i = 0; i < num_work_groups * num_devices; i++){
    sum += partial_sum[i];
  }

  end_time = get_time();
  printf("Elapsed time: %f sec\n", end_time-start_time);

  for(i=0; i<num_devices; i++){
    clReleaseMemObject(buf_partial_sum[i]);
  }
  free(partial_sum);
  for(i=0; i<num_devices; i++){
    clReleaseKernel(kernel[i]);
  }
  for(i=0; i<num_devices; i++){
    clReleaseCommandQueue(queue[i]);
  }
  clReleaseContext(context);

    // return (double)sum / N;
  return sum;

}
