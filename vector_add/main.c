#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

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

    int *A = (int) malloc(sizeof(int) * 16384);
    int *B = (int) malloc(sizeof(int) * 16384);
    int *C = (int) malloc(sizeof(int) * 16384);

    for (i = 0; i < 16384; i++) { // 초기화
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
    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 16384, NULL, &err); //버퍼 오브젝트 만든다. int값 16384개를 저장할 버퍼 오브젝트를 만든다.
    CHECK_ERROR(err);

    // clCreateBuffer(cl_context   /* context */,
    //                cl_mem_flags /* flags */,     // 여러가지 flag를 줄 수있다. CL_MEM_READ_WRITE, MEM_READ_ONLY
    //                size_t       /* size */,
    //                void *       /* host_ptr */,
    //                cl_int *     /* errcode_ret */)

    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 16384, NULL, &err);
    CHECK_ERROR(err);
    bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * 16384, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, bufA, CL_FALSE, 0, sizeof(int) * 16384, A, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, bufB, CL_FALSE, 0, sizeof(int) * 16384, B, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    CHECK_ERROR(err);

    size_t global_size = 16384;
    size_t local_size = 256;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(int) * 16384, C, 0, NULL, NULL);
    CHECK_ERROR(err);

    for (i = 0; i < 16384; i++) {
        if (A[i] + B[i] != C[i]) {
            printf("Verification failed! A[%d] = %d, B[%d] = %d , C[%d] = %d", i, A[i], i, B[i], i, C[i]);
            break;
        }
    }
    else (i == 16384)
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
