#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <unistd.h>

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

double get_time(); // use the get_time() function in mat_mul.c

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

void mat_mul_opencl(float *A, float *B, float *C,
                    int ROW_A, int COL_A, int COL_B) {
  // TODO
  cl_platform_id platform;

    cl_device_id device;

    cl_context context;

    cl_command_queue queue;

    cl_program program;

    char *kernel_source;

    size_t kernel_source_size;

    cl_kernel kernel;

    cl_int err;

    int i;

    err = clGetPlatformIDs(1, &platform, NULL);

    CHECK_ERROR(err);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);

    kernel_source = get_source_code("kernel.cl", &kernel_source_size);
    program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, &kernel_source_size, &err);  //kernel source를 가지고 프로그램을 만든다.

    CHECK_ERROR(err);

    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size;
        char *log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        CHECK_ERROR(err);

        log = (char *) malloc(log_size + 1);

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    }
    CHECK_ERROR(err);

    kernel = clCreateKernel(program, "mat_mul", &err);
    CHECK_ERROR(err);

    cl_mem bufA, bufB, bufC;  // 버퍼를 만든다.

    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*ROW_A*COL_A, NULL, &err);  // A크기만큼의 버퍼 bufA를 만든다.
    CHECK_ERROR(err);
    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*COL_A*COL_B, NULL, &err);  //  B크기만큼
    CHECK_ERROR(err);
    bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*ROW_A*COL_B, NULL, &err); // C크기만큼 C에서 디바이스에서 쓸 꺼니 CL_MEM_READ_WRITE로 한다
    CHECK_ERROR(err);

    double start_time = get_time();  // Command Queue에 

    err = clEnqueueWriteBuffer(queue, bufA, CL_FALSE, 0, sizeof(float)*ROW_A*COL_A, A, 0, NULL, NULL); //Enqueue commands to write to a buffer object from host memory. 
    CHECK_ERROR(err);   // host memory에 있는 A를 bufA에 집어 넣는다. 

    err = clEnqueueWriteBuffer(queue, bufB, CL_FALSE, 0, sizeof(float)*COL_A*COL_B, B, 0, NULL, NULL); // host B -> Device BufB
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);   //kernel.cl 에 있는 인자 6개를 참조하면 된다.
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 3, sizeof(cl_int), &ROW_A);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 4, sizeof(cl_int), &COL_A);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 5, sizeof(cl_int), &COL_B);
    CHECK_ERROR(err);

    size_t global_size[2] = {COL_B, ROW_A};   // column, row 순서이다 헷갈리지 마시 
    size_t local_size[2] = {16, 16};          // 워크-그룹 크기이다.

    global_size[0] = (global_size[0]+local_size[0]-1)/local_size[0]*local_size[0];  // 워크-그룹에 사이즈에 맞게 전체 크기를 맞춰주는 작업이다 .별거없음
    global_size[1] = (global_size[1]+local_size[1]-1)/local_size[1]*local_size[1];  // local_size의 배수가 되게 맞추는 과정이다.
 
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL); ///* num_events_in_wait_list */, -> 0
    CHECK_ERROR(err);

    // clEnqueueNDRangeKernel(cl_command_queue /* command_queue */,
    //                    cl_kernel        /* kernel */,
    //                    cl_uint          /* work_dim */,
    //                    const size_t *   /* global_work_offset */,
    //                    const size_t *   /* global_work_size */,
    //                    const size_t *   /* local_work_size */,
    //                    cl_uint          /* num_events_in_wait_list */,
    //                    const cl_event * /* event_wait_list */,
    //                    cl_event *       /* event */)

    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float)*ROW_A*COL_B, C, 0, NULL, NULL);
    CHECK_ERROR(err);

    // clEnqueueReadBuffer(cl_command_queue    /* command_queue */,
    //                 cl_mem              /* buffer */,
    //                 cl_bool             /* blocking_read */,   // CL_FALSE 나 CL_TRUE이다 기다릴껀지 말껀지 알아보는 옵션
    //                 size_t              /* offset */,
    //                 size_t              /* size */, 
    //                 void *              /* ptr */,
    //                 cl_uint             /* num_events_in_wait_list */,
    //                 const cl_event *    /* event_wait_list */,
    //                 cl_event *          /* event */)

    double end_time = get_time();

    printf("Elapsed time (excl. initialization): %f sec\n", end_time - start_time);

    clReleaseMemObject(bufA);  // 객체가 저장된 메모리가 해제
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
