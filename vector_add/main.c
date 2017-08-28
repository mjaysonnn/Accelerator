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

char *get_source_code(const char *file_name, size_t *len) {
    char *source_code;
    size_t length;
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    length = (size_t) ftell(file);
    rewind(file);

    source_code = (char *) malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';

    fclose(file);

    *len = length;
    return source_code;
}


double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

int main() {
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
	cl_event kernel_event;
	cl_ulong time_start, time_end;

    printf("Program : C[163840000] = A[163840000] + B[163840000]\n\n");

    err = clGetPlatformIDs(1, &platform, NULL);

    CHECK_ERROR(err);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
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

        log = (char *) malloc(log_size + 1);

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        CHECK_ERROR(err);

        log[log_size] = '\0';
        printf("Compiler error:\n%s\n", log);
        free(log);
        exit(0);
    }
    CHECK_ERROR(err);

    kernel = clCreateKernel(program, "vec_add", &err);
    CHECK_ERROR(err);

    // clCreateKernel(cl_program      /* program */,
    //                const char *    /* kernel_name */,
    //                cl_int *        /* errcode_ret */)

    int *A = (int*) malloc(sizeof(int) * 163840000);
    int *B = (int*) malloc(sizeof(int) * 163840000);
    int *C = (int*) malloc(sizeof(int) * 163840000);

    for (i = 0; i < 163840000; i++) { // 초기화
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    cl_mem bufA, bufB, bufC;   //버퍼 오브젝트를 만들기 위한 과정.
    /*
    1. 호스트가 메모리 오브젝트를 만든다
    2. 그리고 호스트 프로그램에서 메모리 전송 커맨드를 이용하면 버퍼 오브젝트에서 이 데이터를 쓴다.
    3. 버퍼 오브젝트에서 커널 함수에게 실행될 때 포인터를 인자로 보낸다.
    4. 커널 함수에서 그러면 일반적인 배열을 쓰듯이 값을 일고 쓴다.
    5. 마지막으로 버퍼 오브젝트는 메모리 전송 커맨드를 이용해 호스트 프로그램이 데이터를 읽게 한다.
    */
    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 163840000, NULL, &err);  CHECK_ERROR(err);

    // clCreateBuffer(cl_context   /* context */,
    //                cl_mem_flags /* flags */,     // 여러가지 flag를 줄 수있다. CL_MEM_READ_WRITE, MEM_READ_ONLY
    //                size_t       /* size */,
    //                void *       /* host_ptr */,
    //                cl_int *     /* errcode_ret */)

    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 163840000, NULL, &err); CHECK_ERROR(err);
    
    bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * 163840000, NULL, &err); CHECK_ERROR(err);

    

    double start0 = get_time();

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sizeof(int) * 163840000, A, 0, NULL, NULL); CHECK_ERROR(err);

    double end0 = get_time();
    printf("bufA -> Device: %f sec\n\n", end0 - start0);

    
    double start1 = get_time();

    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, sizeof(int) * 163840000, B, 0, NULL, NULL); CHECK_ERROR(err);

    double end1 = get_time();
    printf("bufB -> Device: %f sec\n\n", end1 - start1);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA); CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB); CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC); CHECK_ERROR(err);

    size_t global_size = 163840000;
    size_t local_size = 32;

    
    double start2 = get_time();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &kernel_event); CHECK_ERROR(err);
	err = clFinish(queue); CHECK_ERROR(err);
	//err = clFlush(queue); CHECK_ERROR(err);
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);

	printf("Kernel Execution Time: %lu ns\n\n", time_end - time_start);
   
	//double end2 = get_time();
    //printf("Kernel Execution: %f sec\n\n", end2 - start2);
    

    double start3 = get_time();
    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(int) * 163840000, C, 0, NULL, NULL); CHECK_ERROR(err);
    //err = clFinish(queue);
	double end3 = get_time();
    printf("BUFC -> C: %f sec\n\n", end3 - start3);

    double start4 = get_time();
    err = clFinish(queue);
    double end4 = get_time();
    printf("CL_FINISH: %f sec\n\n", end4 - start4);

    

    for (i = 0; i < 163840000; i++) {
        if (A[i] + B[i] != C[i]) {
            printf("Verification failed! A[%d] = %d, B[%d] = %d , C[%d] = %d", i, A[i], i, B[i], i, C[i]);
            break;
        }
    }
    if (i == 163840000)
    {
        printf("Verification success!\n");
    }


    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    free(A);
    free(B);
    free(C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
