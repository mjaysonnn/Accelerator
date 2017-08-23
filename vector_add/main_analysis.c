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

    int *A = (int*) malloc(sizeof(int) * 16384);
    int *B = (int*) malloc(sizeof(int) * 16384);
    int *C = (int*) malloc(sizeof(int) * 16384);

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
    bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 16384, NULL, &err); //커널에서 읽기만 하는 A를 위한 bufA 버퍼 오브젝트를 만든다.
    CHECK_ERROR(err);

    // clCreateBuffer(cl_context   /* context */,
    //                cl_mem_flags /* flags */,
    //                size_t       /* size */,
    //                void *       /* host_ptr */,
    //                cl_int *     /* errcode_ret */)

    /* cl_mem_flags and cl_svm_mem_flags - bitfield */
    // #define CL_MEM_READ_WRITE                           (1 << 0)
    // #define CL_MEM_WRITE_ONLY                           (1 << 1)
    // #define CL_MEM_READ_ONLY                            (1 << 2)
    // #define CL_MEM_USE_HOST_PTR                         (1 << 3)
    // #define CL_MEM_ALLOC_HOST_PTR                       (1 << 4)
    // #define CL_MEM_COPY_HOST_PTR                        (1 << 5)

    bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 16384, NULL, &err);  //커널에서 읽기만 하는 bufB 오브젝트를 만든다.
    CHECK_ERROR(err);

    bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * 16384, NULL, &err);  //커널에서 읽기만 하는 bufC 오브젝트를 만든다.
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, bufA, CL_FALSE, 0, sizeof(int) * 16384, A, 0, NULL, NULL); // 버퍼 쓰기다. 호스트 메모리에 있는 A를 디바이스의 글로벌 메모리로 옮겨준다
    CHECK_ERROR(err);

    // clEnqueueWriteBuffer(cl_command_queue   /* command_queue */,  //디바이스에 커맨드를 보내기 위해 필요하
    //                      cl_mem             /* buffer */,            // 버퍼 오브젝트
    //                      cl_bool            /* blocking_write */,    // 동기화이다. 버퍼 쓰기가 완료된 다음에 Return 하겠냐 안하겠냐이다. false는 커맨드가 커맨드-큐에 enqueue되자마자 return
    //                      size_t             /* offset */,            // 위치
    //                      size_t             /* size */,              // 얼마만큼의 사이즈의 메모리인지
    //                      const void *       /* ptr */,               // 포인터 , 즉 호스트 메모리의 A를 뜻한다.
    //                      cl_uint            /* num_events_in_wait_list */,    //
    //                      const cl_event *   /* event_wait_list */, 
    //                      cl_event *         /* event */)

    err = clEnqueueWriteBuffer(queue, bufB, CL_FALSE, 0, sizeof(int) * 16384, B, 0, NULL, NULL);  // 메인 메모리의 B 배열의 데이터를 버퍼 bufB에 쓴다.
    CHECK_ERROR(err);

    /* 
    커널은 커널 안에서 버퍼에 접근을한다.
    
    * 메모리 오브젝트는 추상적인 메모리 영역이다 *

    특정 디바이스의 메모리에 dedicate된 것이 아님
    실제로 디바이스에서 사용될 때 비로소 글로벌 메모리에 할당 저장된다.( 해당 디바이스의 커맨드-큐에 커맨드를 넣어서 실행할때 디바이스를 찾아준다) WriteBuffer할때 디바이스로 가준다.
    여러 디바이스에서 동시에 할당 될 수 있다.

    * OpenCL 커널 함수 *

    같은 커널 코드를 다른 데이터 아이템에 동시에 실행한다.
    데이터 병렬성을 활용한다

    * 커널 실행 *

    호스트 프로그램에서 커널을 실행시킬 때 N-차원의 인덱스 공간을 지정한다
    워크-아이템 -> 인덱스 공간의 각 점마다 커널 인스턴스가 하나씩 실행된다
                커널 함수 안에서 빌트인 함수- get_global_id 를 사용해 인덱스 공간에서의 워크-아이템 ID를 얻어올 수 있다.

    워크-그룹 -> 여러 워크-아이템들은 워크-그룹으로 묶여 있다.
               모든 워크-그룹의 크기는 동일하다

    하나의 워크-그룹은 하나의 CU에서 실행된다.
            -> 워크-그룹 안의 여러 워크-아이템들이 CU 안의 여러 PE에서 나뉘어 실행한다
            -> 같은 워크-그룹 안의 워크-아이템들은 로컬 메모리를 공유
            -> 워크-그룹 크기가 성능에 큰 영향을 미침한다.
    */
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);   // 커널 인자 설정 해주는 함수이다. 버퍼 A를 0번 인자로 넘긴다.
    CHECK_ERROR(err);

    // clSetKernelArg(cl_kernel    /* kernel */,
    //            cl_uint      /* arg_index */,
    //            size_t       /* arg_size */,
    //            const void * /* arg_value */)

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);     // 커널 인자 설정 해주는 함수이다. 버퍼 B를 1번 인자로 넘긴다.
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);     // 커널 인자 설정 해주는 함수이다. 버퍼 C를 2번 인자로 넘긴다.
    CHECK_ERROR(err);

    size_t global_size = 16384;
    size_t local_size = 256;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);  // 커널을 실행한다 1차원 인덱스의 공간을 만든다.
    CHECK_ERROR(err);                                                                                // // 여러 커널 인스턴스가 이렇게 동시에 실행된다.

    /* 커널 실행할때 인덱스 공간을 보자면 워크-그룹 단위로 CU에 할당해준다 그러면 워크-아이템들이 여러 PE에서 나누어 실행된다. 실제 GPU는 SM이 CU이고 ALU(연산처리장치)가 PE(Processing Element)이다. 
        GPU는 대부분 한 SM(CU)안에 ALU(PE)가 32개에서 64개이다. */

    // global_work_size[i]는 local_work_size[i] 로 항상 나누어 떨어져야 함
    // Local_work_size를 NULL로 지정하면 런타임에서 알아서 워크 그룹 크기를 결정해준다.


    // clEnqueueNDRangeKernel(cl_command_queue /* command_queue */,
    //                    cl_kernel        /* kernel */,
    //                    cl_uint          /* work_dim */,    // 몇 차원 공간인지 나타내준다.
    //                    const size_t *   /* global_work_offset */,  //NULL로 한다.
    //                    const size_t *   /* global_work_size */,    //전체-워크-아이템 개수 지정해준다.
    //                    const size_t *   /* local_work_size */,     // 한 워크-그룹 안의 워크-아이템 개수 지저애준다.
    //                    cl_uint          /* num_events_in_wait_list */, // 동기화이다. 생략
    //                    const cl_event * /* event_wait_list */,          // NULL
    //                    cl_event *       /* event */)                     //  NULL

    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(int) * 16384, C, 0, NULL, NULL); //디바이스의 글로멀 메모리는 호스트 메모리로 보내 읽는 과정이다. 버퍼 bufC의 데이터를 메인 메모리의 C배열로 읽어 출력
    CHECK_ERROR(err);                                                                           

    // clEnqueueReadBuffer(cl_command_queue    /* command_queue */,
    //                 cl_mem              /* buffer */,
    //                 cl_bool             /* blocking_read */,
    //                 size_t              /* offset */,
    //                 size_t              /* size */, 
    //                 void *              /* ptr */,
    //                 cl_uint             /* num_events_in_wait_list */,
    //                 const cl_event *    /* event_wait_list */,   //NULL 로 한다.
    //                 cl_event *          /* event */)             //NULL

    for (i = 0; i < 16384; i++) {
        if (A[i] + B[i] != C[i]) {
            printf("Verification failed! A[%d] = %d, B[%d] = %d , C[%d] = %d", i, A[i], i, B[i], i, C[i]);
            break;
        }
    }
    if (i == 16384)
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
