#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include "bmpfuncs.h"

static float theta = 3.14159/6;

void rotate(float *input_image, float *output_image, int image_width, int image_height,  // 구현해야 할 함수   
            float sin_theta, float cos_theta);

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <src file> <dest file>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  float sin_theta = sinf(theta);
  float cos_theta = cosf(theta);  // theta 만큼 돌리겠다.

  int image_width, image_height;  // width랑 height가 주어지고 
  float *input_image = readImage(argv[1], &image_width, &image_height); //그에 해당하는 이미지를 읽는다.
  float *output_image = (float*)malloc(sizeof(float) * image_width * image_height); // output_image도 메모리 할당을 한 다음에
  rotate(input_image, output_image, image_width, image_height, sin_theta, cos_theta); //rotate함수를 이용하여 회전시킨다음에
  storeImage(output_image, argv[2], image_height, image_width, argv[1]); // 결과값을 저장한다.
  return 0;
}

#define CHECK_ERROR(err) \    // 에러 체크하는 함수  
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

char *get_source_code(const char *file_name, size_t *len) {  // 커널을 읽어서 하나의 문자열로 만든다.
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

void rotate(float *input_image, float *output_image, int image_width, int image_height,
            float sin_theta, float cos_theta) {
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

    err = clGetPlatformIDs(1, &platform, NULL);  // 플랫폼 ID 1개만 얻어오기

    CHECK_ERROR(err);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL); // GPU 디바이스 ID 아무거나 1개 얻어오기
    CHECK_ERROR(err);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);  // 디바이스 1개를 사용하는 컨텍스트 만들기
    CHECK_ERROR(err);

    // clCreateContext(const cl_context_properties * /* properties */,  //별거 아님 NULL로 하면 된다고 함
    //             cl_uint                 /* num_devices */,
    //             const cl_device_id *    /* devices */,
    //             void (CL_CALLBACK * /* pfn_notify */)(const char *, const void *, size_t, void *),
    //             void *                  /* user_data */,
    //             cl_int *                /* errcode_ret */)

    queue = clCreateCommandQueue(context, device, 0, &err);  // In-Order Comman Queue를 만든다. 0 이 In-Order 이다.
    CHECK_ERROR(err);

    kernel_source = get_source_code("kernel.cl", &kernel_source_size);  // kernel을 읽어서 하나의 문자열로 
    program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, &kernel_source_size, &err); //하나의 문자열인 스트링 1개로 프로그램 오브젝트를 만들겠다.

    CHECK_ERROR(err);

    err = clBuildProgram(program, 1, &device, "", NULL, NULL); // 프로그램 오브젝트를 가지고 프로그램을 빌드 해준다. 내부적으로 OpenCL 프레임워크의 OpenCL C 컴파일러 호출해준다. 디바이스 1개에 대해 프로그램을 빌드한다.
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

    kernel = clCreateKernel(program, "img_rotate", &err);  //커널 오브젝트 만들기 img_rotate라는 이름을 가진 커널 함수에 대한 커널 오브젝트 만들기  
    CHECK_ERROR(err);

    size_t image_size = sizeof(float)*image_width*image_height;  //image_size width*height만큼 할당해준다.
    cl_mem src, dest;

    src= clCreateBuffer(context, CL_MEM_READ_ONLY, image_size, NULL, &err); // float 값 width*height 만큼을 저장할 버퍼 오브젝트를 만든다. src 커널에서는 읽기만 하는 것이다.
    CHECK_ERROR(err);

    // 5번째 인자는 버퍼 오브젝트를 host_ptr이 가리키는 곳의 데이터로 초기화해준다. 

    dest = clCreateBuffer(context, CL_MEM_READ_WRITE, image_size, NULL, &err); // float 값 width*height 만큼을 저장할 버퍼 오브젝트를 만든다. dest는 만들고 따로 host_memory에서 쓰지 않는다. 나중에 읽을때 쓴
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, src, CL_FALSE, 0, image_size, input_image, 0, NULL, NULL); // 버퍼 쓰기 , 메인 메모리 input_image 를 데이터 src에 써준다.
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dest);  // 커널 함수의 인자 값 0번 인자 1번 인자 ..
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &src);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_int), &image_width);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_int), &image_height);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_float), &sin_theta);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(cl_float), &cos_theta);
    CHECK_ERROR(err);

    size_t global_size[2] = {image_width, image_height};  //global_size , 워크-아이템 전체 크기
    size_t local_size[2] = {16, 16};  //워크-그룹의 크기
    global_size[0] =(global_size[0]+local_size[0]-1)/local_size[0]*local_size[0]; // 워크-그룹에 맞게 global_size를 맞춘다.
    global_size[1] =(global_size[1]+local_size[1]-1)/local_size[1]*local_size[1]; // 워크-그룹에 맞게 global_size를 맞춘다.

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL); // 커널 실행 2차원 Dimension
    CHECK_ERROR(err);

    err = clEnqueueReadBuffer(queue, dest, CL_TRUE, 0, image_size, output_image, 0, NULL, NULL); // dest라는 버퍼를 메인 메모리의 output_image로 읽어와 출력한다.
    CHECK_ERROR(err);

    clReleaseMemObject(src);
    clReleaseMemObject(dest);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
