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
  cl_command_queue queue[2];  // Command Queue를 2개 쓰기 위해서이다. 하나는 커널을 실행하고 하나는 데이터 전송(부분합 받음)을 위함이다.
  cl_program program;
  char *kernel_source;
  size_t kernel_source_size;
  cl_kernel kernel;
  cl_int err;

  err = clGetPlatformIDs(1, &platform, NULL); CHECK_ERROR(err);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL); CHECK_ERROR(err);

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);  CHECK_ERROR(err);

  queue[0] = clCreateCommandQueue(context, device, 0, &err);  CHECK_ERROR(err);  // 커널을 실행하는 Command Queue이다.

  queue[1] = clCreateCommandQueue(context, device, 0, &err);  CHECK_ERROR(err); // 부분합을 받기 위한 Command Queue이다.


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
  start_time = get_time();  //시간 측정해보자

  size_t global_size = N/2; // N을 절반으로 나눌 것이다.
  size_t local_size = 256;  // local_size는 245으로 했다. 더 넓혀도 될듯하다.
  size_t num_work_groups = global_size / local_size; // local_size를 통해 워크-그룹의 개수를 구할 수있다. 부분합을 받을 때 work_group 별로 받을 것이기때문에 work_group의 개수를 받는다.

  cl_mem buf_partial_sum[2];  // 부분합을 받기 위해 Device용 메모리를 만든다.

  buf_partial_sum[0]=clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*num_work_groups, NULL, &err);  CHECK_ERROR(err); //여기에 부분 합을 받을 것이다. 그래서 num_work_groups 만큼의 데이터를 디바이스에 만든다.

  buf_partial_sum[1]=clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double)*num_work_groups, NULL, &err);  CHECK_ERROR(err); // 똑같이 만든다.

  cl_event kernel_event[2];  // kernel_event Command Queue가 2개이다 보니 순서를 지키기 위해 event object를 만든다.
  int base, offset;   

  base=0;
  offset=N/2;   // 처음부터 중간까지만 계산을 하기 위함이다.

  err =clSetKernelArg(kernel, 0, sizeof(int), &N); CHECK_ERROR(err);
  err =clSetKernelArg(kernel, 1, sizeof(int), &base); CHECK_ERROR(err);
  err =clSetKernelArg(kernel, 2, sizeof(int), &offset); CHECK_ERROR(err);
  err =clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_partial_sum[0]); CHECK_ERROR(err);
  err =clSetKernelArg(kernel, 4, sizeof(double)*local_size, NULL); CHECK_ERROR(err);  //local memory를 선언한다. size는 local_size 만큼

  err = clEnqueueNDRangeKernel(queue[0], kernel, 1, NULL, &global_size, &local_size, 0, NULL, &kernel_event[0]); CHECK_ERROR(err);  // queue[0]에서 계산을 수행한다. 마지막에 보면 kernel_event 라는게 있다.
  /*
  event
Returns an event object that identifies this particular kernel execution instance. 
Event objects are unique and can be used to identify a particular kernel execution instance later on. 
If event is NULL, no event will be created for this kernel execution instance and therefore it will not be possible for the application to query or queue a wait for this particular kernel execution instance.
  */
  err = clFlush(queue[0]); CHECK_ERROR(err);  //이게 non-blocking이니 command-queue에서 런타임 시스템의 스케줄러(디바이스)로 보낸다.

  base = N/2;
  offset = N/2; // 중간부터 끝까지 

  err =clSetKernelArg(kernel, 0, sizeof(int), &N); CHECK_ERROR(err);  // 커널을 인자로 받는다.
  err =clSetKernelArg(kernel, 1, sizeof(int), &base); CHECK_ERROR(err); 
  err =clSetKernelArg(kernel, 2, sizeof(int), &offset); CHECK_ERROR(err);
  err =clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_partial_sum[1]); CHECK_ERROR(err);  //마지막 절반을 인자로 한다.
  err =clSetKernelArg(kernel, 4, sizeof(double)*local_size, NULL); CHECK_ERROR(err);

  err = clEnqueueNDRangeKernel(queue[0], kernel, 1, NULL, &global_size, &local_size, 0, NULL, &kernel_event[1]); CHECK_ERROR(err);  //queue[1] 라는 event object를 생성한다. 이것또한 순서때문에 만들었다.
  err = clFlush(queue[0]); CHECK_ERROR(err);  // 커맨드 -큐에서 런타임 시스템(디바이스)으로 보낼떄까지 기다린다. non-blocking이니 더욱 그렇다. 디바이스로 보내지니 실행을 한다. 물론 clFlush됬다고 이 커널은 끝난게 아니다. 실행을 한다고 보면된다.

  double *partial_sum = (double*)malloc(sizeof(double)*num_work_groups); // Host 용 메모리
  double sum=0;
  int i;

  err = clEnqueueReadBuffer(queue[1], buf_partial_sum[0], CL_FALSE, 0, sizeof(double)*num_work_groups, partial_sum, 1, &kernel_event[0], NULL); CHECK_ERROR(err); // kernel_event[0]이 끝나면 이걸 하겠다는 것이다  event_wait_list 는 1개가 있다. 아마 그게 계산이다 이 계산이 끝나고 부분합을 받겠다.
  /*
  kernel_event는 하나라서 상관이 없나보다. 모든 커맨드 큐는 런타임에 이어진다. 계산이 끝나고 버퍼 읽기가 실행된다. queue가 달라도 할 수있다. 왜냐면 호스트 프로그램에서 지정해주기 대문이다. 이벤트를 통해 커맨드 간의 순서 정의가 가능해진다.  wait_list 에 있는 이벤트마가 모드 Complete가 상태일 떄까지 커맨드가 커맨드-큐에서 기다린다.
  */
  err = clFlush(queue[1]); CHECK_ERROR(err);  //디바이스로 보내질때까지 기다린다.
  err = clFinish(queue[1]); CHECK_ERROR(err); // buf_partial_sum들 다 받을때까지 기다린다.

  for (i = 0; i< num_work_groups; i++){
    sum += partial_sum[i];
  }

  err = clEnqueueReadBuffer(queue[1], buf_partial_sum[1], CL_FALSE, 0, sizeof(double)*num_work_groups, partial_sum, 1, &kernel_event[1], NULL); CHECK_ERROR(err); //kernel_event가 2번째 계산인데 이거 Complete하고 이걸 실행한다.
  err = clFlush(queue[1]); CHECK_ERROR(err);
  err = clFinish(queue[2]); CHECK_ERROR(err);
  
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