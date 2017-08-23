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

    err = clGetPlatformIDs(1, &platform, NULL); // Platform Id를 얻어오자
    
    CHECK_ERROR(err);

    // clGetPlatformIDs(cl_uint          /* num_entries */,
                    // cl_platform_id * /* platforms */,
                    // cl_uint *        /* num_platforms */) CL_API_SUFFIX__VERSION_1_0;


    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);  // Device Id를 얻어오자
    
    CHECHK_ERROR(err); 

    // clGetDeviceIDs(cl_platform_id   /* platform */,
    //                cl_device_type   /* device_type */, 
    //                cl_uint          /* num_entries */, 
    //                cl_device_id *   /* devices */, 
    //                cl_uint *        /* num_devices */)

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);   // 컨텍스트(객) 만들기 , 커널이 실행되는 환경을 만들어준다. 다른 객체들을 관리하기 위한 최상위 객체, 컨텍스트 단위로 커맨드 간 동기화 및 메모리 관리를 수     
                                                                     // 디바이스 1개를 사용하는 COntext 만들기
    CHECK_ERROR(err);

    // clCreateContext(const cl_context_properties * /* properties */,  // 무시하면 됩니더
    //                 cl_uint                 /* num_devices */,
    //                 const cl_device_id *    /* devices */,       //해당 컨텍스트 안에서 사용할 디바이스를 지정
    //                 void (CL_CALLBACK * /* pfn_notify */)(const char *, const void *, size_t, void *),
    //                 void *                  /* user_data */,
    //                 cl_int *                /* errcode_ret */)


    queue = clCreateCommandQueue(context, device, 0, &err);  // 디바이스에 커맨드를 보내기 위해 필요한 커맨드 큐를 만든다, 호스트가 커맨드-쿠에 커맨드를 넣으면 OpenCl의 런타임 시스템에서 이것을 빼내어 실행
    CHECK_ERROR(err);                                        // 디바이스마다 커맨드-큐를 만들어줘야한다.  3번째 Parameter의 0은 In-Order를 뜻한다.

    // clCreateCommandQueue(cl_context                     /* context */,
    //                     cl_device_id                   /* device */,
    //                     cl_command_queue_properties    /* properties */,   // In-Order 커맨들이 Enqueue 한대로 순서대로 실행, Out-of-Order enquee 순서와 무관, 이거 잘 안씀
    //                     cl_int *                       /* errcode_ret */) 


    // /* cl_command_queue_properties - bitfield */
    // #define CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE      (1 << 0)   //이걸 쓰면 OutOrder로 된다. 
    // #define CL_QUEUE_PROFILING_ENABLE                   (1 << 1)
    // #define CL_QUEUE_ON_DEVICE                          (1 << 2)
    // #define CL_QUEUE_ON_DEVICE_DEFAULT                  (1 << 3)


    kernel_source = get_source_code("kernel.cl", &kernel_source_size); // kernel.cl이라는 소스 코드를 읽어오는 함수이다.

    program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, &kernel_source_size, &err); // 소스 코드 문자열로부터 프로그램 오브젝트 만들기 , kernel_source로 부터, 스트링 1개 쓰겠따는 의미

    // 프로그램 오브젝트란 소스 코드를 입력으로 준 다음 컴파일에서 OpenCL 프로그램의 바이너리로 만드는 과정

    // clCreateProgramWithSource(cl_context        /* context */,  소스 코드를 파일에서 읽어 들어 프로그램 오브젝트 만들기
    //                           cl_uint           /* count */,    스트링 1개를 쓰겠다는 의미,    
    //                           const char **     /* strings */,
    //                           const size_t *    /* lengths */,
    //                           cl_int *          /* errcode_ret */)

    CHECK_ERROR(err);

    err = clBuildProgram(program, 1, &device, "", NULL, NULL); // 내부적으로 OpenCL 프레임워크의 OpenCL C 컴파일러를 호출한다. 이 컴파일러는 프로그램 오브젝트를 빌드한다고 보면 된다. 컴파일을 수행할 디바이스 리스트를 지정해주어야한다.
                                                               // 디바이스 1개에 대해 프로그램을 빌드해준다.
    // clBuildProgram(cl_program           /* program */,
    //                cl_uint              /* num_devices */,
    //                const cl_device_id * /* device_list */,
    //                const char *         /* options */,    // 옵션이 엄청 많다. 매크로 name을 정의하는 것도 있고, 모든 컴파일러 최적화 끄는 것도 있고 다양하다 (https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clBuildProgram.html)
    //                void (CL_CALLBACK *  /* pfn_notify */)(cl_program /* program */, void * /* user_data */),  // skip
    //                void *               /* user_data */)     //skip


    if (err == CL_BUILD_PROGRAM_FAILURE) {  // 에러 코드 이다. 
        size_t log_size;
        char *log;

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size); // 컴파일 에러 발생 시 쓰는 함수이다 먼저 로그 사이즈를 얻는다 

        // clGetProgramBuildInfo(cl_program            /* program */,
        //                       cl_device_id          /* device */,
        //                       cl_program_build_info /* param_name */,
        //                       size_t                /* param_value_size */,
        //                       void *                /* param_value */,
        //                       size_t *              /* param_value_size_ret */)

        CHECK_ERROR(err);

        log = (char*)malloc(log_size+1);

        err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);  
        //특정 디바이스에 대해서 디바이스마다 각각 컴파일을 수행하므로 CL_PROGRAM_BUILD_LOG로 지정한다.
        // 로그 사이즈 만큼의 로그를 얻는다. 

        CHECK_ERROR(err);

        log[log_size] = '\0';

        printf("Compiler error:\n%s\n", log);

        free(log);

        exit(0);
    }
    CHECK_ERROR(err);

    kernel = clCreatKernel(program, "my_kernel", &err); //커널 오브젝트를 만든다. OpenCL 프로그램은 여러 커널로 이루어짐
                                                        // 디바이스에서 실행되는 코드의 기본 단위이다
                                                        // 여러 커널 인스턴스가 동시에 디바이스에서 실행이 된다.
                                                        // 커널 오브젝트는 아까 만든 프로그램 오브젝트에서 특정 커널 함수만 분리한다.
                                                        // 프로그램이 빌드 된 다음 만들어야한다.(바로 전단계를 의미한다.)
                                                        // 나중에 커널 오브젝트를 사용해 커널을 실행한다.

    // clCreateKernel(cl_program      /* program */,
    //                const char *    /* kernel_name */,
    //                cl_int *        /* errcode_ret */)  

    CHECK_ERROR(err);

    clReleaseKernel(kernel); // Referecne Count 와 관련있다. 객체가 몇 군데서 참조되었는지 나타내 준다. (객체)
    clReleaseProgram(program);  // Reference COunt가 0이 되면 알아서 해당 객체가 저장된 메모리가 해제된다. 매우 중요하다
    clReleaseCommandQueue(queue);   // 객체를 사용하느 중에는 reference count를 0보다 크게 유지해야한다.
    clReleaseContext(context);  // kernel , program(객체들의 집합) , queue(command-queue), context(각 디바이스들에서 사용할 수 있는 queue, buffer 할당) 이 모든게 객체이다.


    printf("Finished!\n");

    return 0;
}