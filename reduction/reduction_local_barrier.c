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

  cl_device_id device;

    cl_context context;

    cl_command_queue queue;

    cl_program program;

    char *kernel_source;

    size_t kernel_source_size;

    cl_kernel kernel;

    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);

    CHECK_ERROR(err);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);

    kernel_source = get_source_code("kernel_local.cl", &kernel_source_size);
    program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, &kernel_source_size, &err);

    CHECK_ERROR(err);

    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char *log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char*)malloc(log_size+1);

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    }
    CHECK_ERROR(err);

    kernel= clCreateKernel(program, "reduction", &err);
    CHECK_ERROR(err);

    size_t global_size = N;
    size_t local_size = 64;
    size_t num_work_groups = global_size / local_size;

    cl_mem buf_array, buf_partial_sum;
    buf_array=clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*N, NULL, &err); // 여기서 cl_mem의 size가 나온다 sizeof(int)*N
    CHECK_ERROR(err);

    buf_partial_sum=clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int)*num_work_groups, NULL, &err); //cl_mem도 사이즈도 나온다 sizeof(int)*num_work_groups
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, buf_array, CL_TRUE, 0, sizeof(int)*N, array, 0, NULL, NULL);
    CHECK_ERROR(err);

    double start_time, end_time;
    start_time = get_time();

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_array);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_partial_sum);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(int)*local_size, NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &N);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    CHECK_ERROR(err);

    int *partial_sum = (int*)malloc(sizeof(int*)*num_work_groups); // 이건 참고로 디바이스에서 만든 데이터이다.
    err = clEnqueueReadBuffer(queue, buf_partial_sum, CL_TRUE, 0, sizeof(int)*num_work_groups, partial_sum, 0, NULL, NULL);
    CHECK_ERROR(err);

    int sum=0;
    int i;
    for(i=0; i<num_work_groups; i++){
      sum += partial_sum[i];
    }
    end_time = get_time();
    printf("Elapsed time: %f sec\n", end_time -start_time);

    clReleaseMemObject(buf_array)
    clReleaseMemObject(buf_partial_sum)
    free(partial_sum);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return (double)sum / N;


}
