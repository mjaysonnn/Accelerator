#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <unistd.h>

static int ROW_A = 10240;
static int COL_A = 10240;
static int COL_B = 10240;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

void mat_mul(float *A, float *B, float *C,
             int ROW_A, int COL_A, int COL_B);
void verify(float *A, float *B, float *C,
            int ROW_A, int COL_A, int COL_B);

int main(int argc, char *argv[]) {
  float *A = (float*)malloc(sizeof(float) * ROW_A * COL_A);
  float *B = (float*)malloc(sizeof(float) * COL_A * COL_B);
  float *C = (float*)malloc(sizeof(float) * ROW_A * COL_B);
  int i, j;

  for (i = 0; i < ROW_A; i++) {
    for (j = 0; j < COL_A; j++) {
      A[i * COL_A + j] = (float)(rand() % 1000) / 100.0f;
    }
  }
  for (i = 0; i < COL_A; i++) {
    for (j = 0; j < COL_B; j++) {
      B[i * COL_B + j] = (float)(rand() % 1000) / 100.0f;
    }
  }

  printf("Matrix Multiplication\n");
  printf("C[%d X %d] = A[%d X %d] X B[%d X %d]\n",
         ROW_A, COL_B, ROW_A, COL_A, COL_A, COL_B);

  mat_mul(A, B, C, ROW_A, COL_A, COL_B);

  //verify(A, B, C, ROW_A, COL_A, COL_B);

  free(A);
  free(B);
  free(C);
  return 0;
}

void verify(float *A, float *B, float *C,
            int ROW_A, int COL_A, int COL_B) {
  int i, j, k;
  float sum;

  for (i = 0; i < ROW_A; i++) {
    for (j = 0; j < COL_B; j++) {
      sum = 0.0f;
      for (k = 0; k < COL_A; k++) {
        sum += A[i * COL_A + k] * B[k * COL_B + j];
      }
      if (fabsf(C[i * COL_B + j] - sum) > 0.1) {
        printf("Verification failed! C[%d][%d]: %f vs. %f\n",
               i, j, C[i * COL_B + j], sum);
        return;
      }
    }
  }
  printf("Verification success!\n");
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

void mat_mul(float *A, float *B, float *C,
             int ROW_A, int COL_A, int COL_B) {
  cl_platform_id platform;
  cl_uint num_devices;
  cl_device_id *device;
  cl_context context;
  cl_command_queue *queue;
  cl_program program;
  char *kernel_source;
  size_t kernel_source_size;
  cl_kernel *kernel;
  int i;
  cl_int err;

  err = clGetPlatformIDs(1, &platform, NULL);   // 플랫폼 ID 아무거나 1개 얻어오기  - 플랫폼은 호스트 + 하나 이상의 디바이스라서 한개만 있으면 된다.
  CHECK_ERROR(err);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);  // 디바이스 개수 확인
  CHECK_ERROR(err);

  printf("%u devices\n", num_devices);

  device = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
  queue = (cl_command_queue*)malloc(sizeof(cl_command_queue) * num_devices);
  kernel = (cl_kernel*)malloc(sizeof(cl_kernel) * num_devices);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, device, NULL);  //  GPU 디바이스 ID 4개 얻어오기
  CHECK_ERROR(err);

  context = clCreateContext(NULL, num_devices, device, NULL, NULL, &err);  // 디바이스 4개를 사용하는 Context 만들기 , Context는 1개만 있어도 된다. 해당 컨텍스트 안에서 사용할 디바이스를 지정한다.
  CHECK_ERROR(err);

  for (i = 0; i < num_devices; i++) {
    queue[i] = clCreateCommandQueue(context, device[i], 0, &err); //디바이스 별로 Command-Queue를 만든다.
    CHECK_ERROR(err);
  }

  kernel_source = get_source_code("kernel.cl", &kernel_source_size);
  program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source,
                                      &kernel_source_size, &err);
  CHECK_ERROR(err);

  err = clBuildProgram(program, num_devices, device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    char *log;

    err = clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG,
                                0, NULL, &log_size);
    CHECK_ERROR(err);

    log = (char*)malloc(log_size + 1);
    err = clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG,
                                log_size, log, NULL);
    CHECK_ERROR(err);

    log[log_size] = '\0';
    printf("Compiler error:\n%s\n", log);
    free(log);
  }
  CHECK_ERROR(err);

  for (i = 0; i < num_devices; i++) {
    kernel[i] = clCreateKernel(program, "mat_mul", &err);  //디바이스 별로 커널 오브젝트를 만든다.
    CHECK_ERROR(err);
  }

  double start_time = get_time();

  int ROW_A_PER_DEVICE = ROW_A / num_devices; // A를 4래로 쪼갠다.

  cl_mem *bufA, *bufB, *bufC;
  bufA = (cl_mem*)malloc(sizeof(cl_mem) * num_devices);
  bufB = (cl_mem*)malloc(sizeof(cl_mem) * num_devices); // 일단 크게 만드는 거 같다.      
  bufC = (cl_mem*)malloc(sizeof(cl_mem) * num_devices);

  for (i = 0; i < num_devices; i++) {         //각각 디바이스별로 bufA, bufB, bufC를 보내준다.
    bufA[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*ROW_A_PER_DEVICE*COL_A,
                             NULL, &err);
    CHECK_ERROR(err);
    bufB[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*COL_A*COL_B,
                             NULL, &err);
    CHECK_ERROR(err);
    bufC[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*ROW_A_PER_DEVICE*COL_B,
                             NULL, &err);
    CHECK_ERROR(err);
  }

  for (i = 0; i < num_devices; i++) {         //디바이스별로 A행렬/4 , B를 bufA, bufB에 써준다.
    err = clEnqueueWriteBuffer(queue[i], bufA[i], CL_FALSE, 0,
                               sizeof(float)*ROW_A_PER_DEVICE*COL_A, A + (ROW_A_PER_DEVICE*COL_A*i),
                               0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[i], bufB[i], CL_FALSE, 0,
                               sizeof(float)*COL_A*COL_B, B,
                               0, NULL, NULL);
    CHECK_ERROR(err);
  }

  for (i = 0; i < num_devices; i++) {   //디바이스별로 커널인자를 만들어준다. 확실히 커널은 디바이스별로 존재 Command Queue와 함    
    err = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), &bufA[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 1, sizeof(cl_mem), &bufB[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 2, sizeof(cl_mem), &bufC[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 3, sizeof(cl_int), &ROW_A_PER_DEVICE);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 4, sizeof(cl_int), &COL_A);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 5, sizeof(cl_int), &COL_B);
    CHECK_ERROR(err);
  }

  size_t global_size[2] = {COL_B, ROW_A_PER_DEVICE};
  size_t local_size[2] = {16, 16};
  global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
  global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];

  for (i = 0; i < num_devices; i++) {             //디바이스 별로 커널을 실행한다
    err = clEnqueueNDRangeKernel(queue[i], kernel[i], 2, NULL, global_size, local_size,
                                 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  for (i = 0; i < num_devices; i++) {   //연산이 끝나고 bufC[i]결과값을 메인 메모리의 C배열로 읽어와 출력한다. 
    err = clEnqueueReadBuffer(queue[i], bufC[i], CL_FALSE, 0,
                              sizeof(float)*ROW_A_PER_DEVICE*COL_B, C + (ROW_A_PER_DEVICE*COL_A*i),
                              0, NULL, NULL);
    CHECK_ERROR(err);
  }

  for (i = 0; i < num_devices; i++) {   // 모든 커맨드 큐에 있는 커맨드가 끝날때까지 기다린다.
    clFinish(queue[i]);
  }

  double end_time = get_time();
  printf("Elapsed time: %f sec\n", end_time - start_time);

  for (i = 0; i < num_devices; i++) {     // 버퍼 오브젝트 디바이스별로 삭제 
    clReleaseMemObject(bufA[i]);
    clReleaseMemObject(bufB[i]);
    clReleaseMemObject(bufC[i]);
  }
  free(bufA);  //메인메모리 에 있는 bufA들을 삭제
  free(bufB);
  free(bufC);
  for (i = 0; i < num_devices; i++) {
    clReleaseKernel(kernel[i]);  //디바이스별로 커널 오브젝트 삭제  
  }
  free(kernel);
  clReleaseProgram(program);  //프로그램은 하나라서 삭제
  for (i = 0; i < num_devices; i++) {
    clReleaseCommandQueue(queue[i]);
  }
  free(queue);  
  clReleaseContext(context); //context device에 있으니 메모리 할당 제거  
  free(device); //평소 device의 4배로 받아서 삭제한다.
}
