#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <unistd.h>

static int N = 536870912;

double get_time() { //  시간 구하는 함수
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

double f(double x){   // 함수 선언
  return (3*x*x + 2*x +1);
}

double integral_seq(int N); // 함수 선언
double integral_opencl(int N);  // 함수 선언

int main() {


  double ans_seq, ans_opencl;
  int i;
  //for (i = 0; i < N; i++) {
  //  array[i] = rand() % 100;
  //}

  printf("Sequential version...\n");
  ans_seq = integral_seq(N);
  printf("int_o^1000 f(x) dx = %f\n", ans_seq);

  printf("OpenCL version...\n");
  ans_opencl = integral_opencl(N);
  printf("int_o^1000 f(x) dx = %f\n", ans_opencl);


  return 0;
}

double integral_seq(int N) { //순차적으로 합을 구한다.
  double dx = (1000.0 / (double)N);
  double sum = 0;
  int i;
  double start_time, end_time;
  start_time = get_time();
  for(i = 0; i< N; i++){
    sum += f(i * dx) * dx;
  }
  end_time = get_time();
  printf("Elapsed time: %f sec\n", end_time-start_time);
  return sum;
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

double integral_opencl(int N) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue[2];  // Command Queue를 2개 쓰기 위해서이다.
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

  queue[0] = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR(err);

  queue[1] = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR(err);


  kernel_source = get_source_code("kernel.cl", &kernel_source_size);
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

  kernel= clCreateKernel(program, "integral", &err);  // reduction이라는 커널 오브젝트 만든다.
  CHECK_ERROR(err);

  double start_time, end_time;
  start_time = get_time();

  size_t global_size = N/2;
  size_t local_size = 256;
  size_t num_work_groups = global_size / local_size; // local_size를 통해 워크-그룹의 개수를 구할 수있다.

  cl_mem buf_partial_sum[2];  // 아무 데이터 타입을 가질수 있는거 같다.

  buf_partial_sum[0]=clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*num_work_groups, NULL, &err);  CHECK_ERROR(err);

  buf_partial_sum[1]=clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*num_work_groups, NULL, &err);  CHECK_ERROR(err);

  cl_event kernel_event[2];
  int base, offset;

  base=0;
  offset=N/2;

  err =clSetKernelArg(kernel, 0, sizeof(int), &N); CHECK_ERROR(err);
  err =clSetKernelArg(kernel, 1, sizeof(int), &base); CHECK_ERROR(err);
  err =clSetKernelArg(kernel, 2, sizeof(int), &offset); CHECK_ERROR(err);
  err =clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_partial_sum[0]); CHECK_ERROR(err);
  err =clSetKernelArg(kernel, 4, sizeof(double)*local_size, NULL); CHECK_ERROR(err);

  err = clEnqueueNDRangeKernel(queue[0], kernel, 1, NULL, &global_size, &local_size, 0, NULL, &kernel_event[0]); CHECK_ERROR(err);
  err = clFlush(queue[0]); CHECK_ERROR(err);

  base = N/2;
  offset = N/2;

  err =clSetKernelArg(kernel, 0, sizeof(int), &N); CHECK_ERROR(err);
  err =clSetKernelArg(kernel, 1, sizeof(int), &base); CHECK_ERROR(err);
  err =clSetKernelArg(kernel, 2, sizeof(int), &offset); CHECK_ERROR(err);
  err =clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_partial_sum[1]); CHECK_ERROR(err);
  err =clSetKernelArg(kernel, 4, sizeof(double)*local_size, NULL); CHECK_ERROR(err);

  err = clEnqueueNDRangeKernel(queue[0], kernel, 1, NULL, &global_size, &local_size, 0, NULL, &kernel_event[1]); CHECK_ERROR(err);
  err = clFlush(queue[0]); CHECK_ERROR(err);

  double *partial_sum = (double*)malloc(sizeof(double)*num_work_groups);
  double sum=0;
  int i;

  err = clEnqueueReadBuffer(queue[1], buf_partial_sum[0], CL_FALSE, 0, sizeof(double)*num_work_groups, partial_sum, 1, &kernel_event[0], NULL); CHECK_ERROR(err);
  err = clFlush(queue[1]); CHECK_ERROR(err);
  err = clFinish(queue[1]); CHECK_ERROR(err);

  for (i = 0; i< num_work_groups; i++){
    sum += partial_sum[i];
  }

  err = clEnqueueReadBuffer(queue[1], buf_partial_sum[1], CL_FALSE, 0, sizeof(double)*num_work_groups, partial_sum, 1, &kernel_event[1], NULL); CHECK_ERROR(err);
  err = clFlush(queue[1]); CHECK_ERROR(err);

  err = clFinish(queue[0]); CHECK_ERROR(err);
  err = clFinish(queue[1]); CHECK_ERROR(err);

  for(i = 0; i< num_work_groups; i++){
    sum += partial_sum[i];
  }

    end_time = get_time();
    printf("Elapsed time: %f sec\n", end_time-start_time);

    clReleaseMemObject(buf_partial_sum[0]);
    clReleaseMemObject(buf_partial_sum[1]);
    // clReleaseMemObject(buf_sum);
    free(partial_sum);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue[0]);
    clReleaseCommandQueue(queue[1]);
    clReleaseContext(context);

    // return (double)sum / N;
    return sum;

}