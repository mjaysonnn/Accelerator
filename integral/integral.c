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

    size_t global_size = N;
    size_t local_size = 256;
    size_t num_work_groups = global_size / local_size; // local_size를 통해 워크-그룹의 개수를 구할 수있다.

    cl_mem buf_partial_sum;  // 아무 데이터 타입을 가질수 있는거 같다.

    buf_partial_sum=clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*num_work_groups, NULL, &err); // 여기서 cl_mem의 size가 나온다 sizeof(int)*N

    CHECK_ERROR(err);

    // buf_sum=clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &err); //cl_mem도 사이즈도 나온다 sizeof(int)*num_work_groups

    // CHECK_ERROR(err);

    // err = clEnqueueWriteBuffer(queue, buf_array, CL_TRUE, 0, sizeof(int)*N, array, 0, NULL, NULL); // N개의 배열인 데이터를 buf_array에 써준다. atomic 연산이니까 CL_TRUE로 해준다.

    // CHECK_ERROR(err);

    // double start_time, end_time;
    // start_time = get_time();

    // int sum = 0;  // sum은 0으로 초기화되어있다.
    // err = clEnqueueWriteBuffer(queue, buf_sum, CL_FALSE, 0, sizeof(int), &sum, 0, NULL, NULL); // sum 이라는 데이터를 buf_sum에 또 다시 써준다.. 이렇게 buf_sum을 0으로 초기화 된다.

    // CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(int), &N); // N개의 배열  , cl_float A =0.5 라도 &A로 받는다.
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_partial_sum); // 합  buf_sum이랑 관련있다.
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(double)*local_size, NULL); // local_memory를 선언해줬다. 위에서 말한 것처럼 local_size만큼의 크기를 가지고있다.
    CHECK_ERROR(err);
    // err = clSetKernelArg(kernel, 3, sizeof(int), &N);
    // CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL); //1차원 인덱스 공간의다.
    CHECK_ERROR(err);

    // int *partial_sum = (int*)malloc(sizeof(int*)*num_work_groups); // 이건 참고로 디바이스에서 만든 데이터이다.
    // err = clEnqueueReadBuffer(queue, buf_partial_sum, CL_TRUE, 0, sizeof(int)*num_work_groups, partial_sum, 0, NULL, NULL);
    // CHECK_ERROR(err);

    // int sum=0;
    // int i;
    // for(i=0; i<num_work_groups; i++){
    //   sum += partial_sum[i];
    // }
    double *partial_sum = (double*)malloc(sizeof(double)*num_work_groups);

    err = clEnqueueReadBuffer(queue, buf_partial_sum, CL_TRUE, 0, sizeof(double)*num_work_groups, partial_sum, 0, NULL, NULL); // buf_sum을 sum으로 다시 읽어준다. device -> host // 기다려야한다 Command가 끝날때까
    CHECK_ERROR(err);

    // end_time = get_time();
    // printf("Elapsed time: %f sec\n", end_time -start_time);
    double sum =0;
    int i;
    for (i=0; i<num_work_groups; i++){
      sum += partial_sum[i];
    }

    end_time = get_time();
    printf("Elapsed time: %f sec\n", end_time-start_time);

    clReleaseMemObject(buf_partial_sum);
    // clReleaseMemObject(buf_sum);
    free(partial_sum);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // return (double)sum / N;
    return sum;

}