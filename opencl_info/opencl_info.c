	#include <stdio.h>
	#include <stdlib.h>
	#include <CL/cl.h>  //OpenCL을 포함하겠다.

	#define CHECK_ERROR(err) \
if (err != CL_SUCCESS) { \
	printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
	exit(EXIT_FAILURE); \
}
	// 항상 이걸로 오류가 났는지 안 났는지 체크한다고 한다

int main() {
	// TODO
	cl_uint num_platforms; // 플랫폼 개수와 얻어올때 쓰인다.

	cl_platform_id *platforms; // platforms id

	cl_uint num_devices; // number of device

	cl_device_id *devices; // id of devices

	char str[1024]; //strings to use

	cl_device_type device_type; //type of device

	size_t max_work_group_size; //work group size의 max를 쓰기 위한

	// class size_t
	// {
	// private:
	//     ::size_t data_[N];

	// public:
	//     //! \brief Initialize size_t to all 0s
	//     size_t()
	//     {
	//         for( int i = 0; i < N; ++i ) {
	//             data_[i] = 0;
	//         }
	//     }

	//     ::size_t& operator[](int index)
	//     {
	//         return data_[index];
	//     }

	//     const ::size_t& operator[](int index) const
	//     {
	//         return data_[index];
	//     }

	//     //! \brief Conversion operator to T*.
	//     operator ::size_t* ()             { return data_; }

	//     //! \brief Conversion operator to const T*.
	//     operator const ::size_t* () const { return data_; }
	// };

	cl_ulong global_mem_size; //global memory size

	cl_ulong local_mem_size; //local memory size

	cl_ulong max_mem_alloc_size; //maxmimum memory allocation size

	cl_uint p, d;

	cl_uint err; //error to check if it's going right




	err = clGetPlatformIDs(0, NULL, &num_platforms); // 플랫폼 개수를  확인할 수 있다.

	//
	//clGetPlatformIDs(cl_uint          /* num_entries */
	//                 cl_platform_id * /* platforms */,
	//                 cl_uint *        /* num_platforms */
	//

	CHECK_ERROR(err); // 오류 확인

	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);  // cl_platform_id 는 자료구조이다. 자료구조 내부적으로 모두 포인터이다. 플롯팸 개수만큼 할당을 해준다.

	err = clGetPlatformIDs(num_platforms, platforms, NULL);  // 플랫폼 ID를 num_platforms개 얻어오기


	//
	//clGetPlatformIDs(cl_uint          /* num_entries */
	//                 cl_platform_id * /* platforms */,
	//                 cl_uint *        /* num_platforms */
	//

	CHECK_ERROR(err);

	printf("Number of platforms: %u\n\n", num_platforms); // platform 의 개수를 출력해준다

	for(p =0; p < num_platforms; p++) { // platform 개수만큼

	printf("platform: %u\n", p); // i번째 platform을 말하구먼

	err = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 1024, str, NULL); // 플랫폼 정보 얻어오기 저장은 str에 된다. 위에 str 정의해 놓음

	//clGetPlatformInfo(cl_platform_id   /* platform */,
	//                  cl_platform_info /* param_name */,    -> param_name에 어떤 정보를 얻을지를 지정해 줌 .  1. CL_PLATFORM_NAME 2. CL_PLATFORM_VENDOR 3.CL_PLATFORM_VERSION 4. CL_PLATFORM_PROFILE  5. CL_PLATFORM_EXTENSIon
	//                  size_t           /* param_value_size */,
	//                  void *           /* param_value */,
	//                  size_t *         /* param_value_size_ret */)

	// /* cl_platform_info */
	// #define CL_PLATFORM_PROFILE                         0x0900
	// #define CL_PLATFORM_VERSION                         0x0901
	// #define CL_PLATFORM_NAME                            0x0902
	// #define CL_PLATFORM_VENDOR                          0x0903
	// #define CL_PLATFORM_EXTENSIONS                      0x0904

	CHECK_ERROR(err);

	printf(" - CL_PLATFORM_NAME : %s\n", str); // Platform 이름을 출력해준다

	err = clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, 1024, str, NULL); // str에 저장되어있음

	//clGetPlatformInfo(cl_platform_id   /* platform */,
	//                  cl_platform_info /* param_name */,
	//                  size_t           /* param_value_size */,
	//                  void *           /* param_value */,
	//                  size_t *         /* param_value_size_ret */)

	CHECK_ERROR(err);

	printf(" - CL_VENDOR_NAME : %s\n\n", str);

	err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices); // 디바이스 개수와 ID 얻어오기

	// clGetDeviceIDs(cl_platform_id   /* platform */,
	//                cl_device_type   /* device_type */,  device_type에 ID를 얻어올 디바이스의 종류를 지정해 줌
	//                cl_uint          /* num_entries */,
	//                cl_device_id *   /* devices */,
	//                cl_uint *        /* num_devices */)

	// /* cl_device_type - bitfield */
	// #define CL_DEVICE_TYPE_DEFAULT                      (1 << 0)
	// #define CL_DEVICE_TYPE_CPU                          (1 << 1)
	// #define CL_DEVICE_TYPE_GPU                          (1 << 2)
	// #define CL_DEVICE_TYPE_ACCELERATOR                  (1 << 3)
	// #define CL_DEVICE_TYPE_CUSTOM                       (1 << 4)
	// #define CL_DEVICE_TYPE_ALL                          0xFFFFFFFF

	CHECK_ERROR(err);

	printf("Number of devices : %u\n\n", num_devices); // 디바이스 개수 출력하기

	devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices); // 디바이스 개수 만큼 디바이스 id 할당해주기

	err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

	// clGetDeviceIDs(cl_platform_id   /* platform */,
	//                cl_device_type   /* device_type */,  device_type에 ID를 얻어올 디바이스의 종류를 지정해 줌
	//                cl_uint          /* num_entries */,
	//                cl_device_id *   /* devices */,
	//                cl_uint *        /* num_devices */)

	// /* cl_device_type - bitfield */
	// #define CL_DEVICE_TYPE_DEFAULT                      (1 << 0)
	// #define CL_DEVICE_TYPE_CPU                          (1 << 1)
	// #define CL_DEVICE_TYPE_GPU                          (1 << 2)
	// #define CL_DEVICE_TYPE_ACCELERATOR                  (1 << 3)
	// #define CL_DEVICE_TYPE_CUSTOM                       (1 << 4)
	// #define CL_DEVICE_TYPE_ALL                          0xFFFFFFFF

	CHECK_ERROR(err);

	for(d=0; d < num_devices; d++){

	printf("device: %u\n", d); // 디바이스의 개수만큼

	err = clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);  //디바이스 정보 얻어오기, Device Type을 출력했다.

	// clGetDeviceInfo(cl_device_id    /* device */,
	//                    cl_device_info  /* param_name */,   cl_device_info 겁나 많다. 출력하고 싶은거 아래보다 더 많음  // global_memory_size 얻고 싶으면 CL_DEVICE_GLOBAL_MEM_SIZE 에 param-value는 &global_mem_size로 한
	//                    size_t          /* param_value_size */,                                               // return type을 알고싶으면 google링 하면 된다.
	//                    void *          /* param_value */,
	//                    size_t *        /* param_value_size_ret */)

	//     /* cl_device_info */
	// #define CL_DEVICE_TYPE                                  0x1000
	// #define CL_DEVICE_VENDOR_ID                             0x1001
	// #define CL_DEVICE_MAX_COMPUTE_UNITS                     0x1002
	// #define CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS              0x1003
	// #define CL_DEVICE_MAX_WORK_GROUP_SIZE                   0x1004
	// #define CL_DEVICE_MAX_WORK_ITEM_SIZES                   0x1005
	// #define CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR           0x1006
	// #define CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT          0x1007
	// #define CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT            0x1008
	// #define CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG           0x1009
	// #define CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT          0x100A
	// #define CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE         0x100B
	// #define CL_DEVICE_MAX_CLOCK_FREQUENCY                   0x100C
	// #define CL_DEVICE_ADDRESS_BITS                          0x100D
	// #define CL_DEVICE_MAX_READ_IMAGE_ARGS                   0x100E
	// #define CL_DEVICE_MAX_WRITE_IMAGE_ARGS                  0x100F
	// #define CL_DEVICE_MAX_MEM_ALLOC_SIZE                    0x1010
	// #define CL_DEVICE_IMAGE2D_MAX_WIDTH                     0x1011
	// #define CL_DEVICE_IMAGE2D_MAX_HEIGHT                    0x1012
	// #define CL_DEVICE_IMAGE3D_MAX_WIDTH                     0x1013
	// #define CL_DEVICE_IMAGE3D_MAX_HEIGHT                    0x1014
	// #define CL_DEVICE_IMAGE3D_MAX_DEPTH                     0x1015
	// #define CL_DEVICE_IMAGE_SUPPORT                         0x1016
	// #define CL_DEVICE_MAX_PARAMETER_SIZE                    0x1017
	// #define CL_DEVICE_MAX_SAMPLERS                          0x1018
	// #define CL_DEVICE_MEM_BASE_ADDR_ALIGN                   0x1019
	// #define CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE              0x101A
	// #define CL_DEVICE_SINGLE_FP_CONFIG                      0x101B
	// #define CL_DEVICE_GLOBAL_MEM_CACHE_TYPE                 0x101C
	// #define CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE             0x101D
	// #define CL_DEVICE_GLOBAL_MEM_CACHE_SIZE                 0x101E
	// #define CL_DEVICE_GLOBAL_MEM_SIZE                       0x101F
	// #define CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE              0x1020
	// #define CL_DEVICE_MAX_CONSTANT_ARGS                     0x1021
	// #define CL_DEVICE_LOCAL_MEM_TYPE                        0x1022
	// #define CL_DEVICE_LOCAL_MEM_SIZE                        0x1023
	// #define CL_DEVICE_ERROR_CORRECTION_SUPPORT              0x1024
	// #define CL_DEVICE_PROFILING_TIMER_RESOLUTION            0x1025
	// #define CL_DEVICE_ENDIAN_LITTLE                         0x1026
	// #define CL_DEVICE_AVAILABLE                             0x1027
	// #define CL_DEVICE_COMPILER_AVAILABLE                    0x1028
	// #define CL_DEVICE_EXECUTION_CAPABILITIES                0x1029
	// #define CL_DEVICE_QUEUE_PROPERTIES                      0x102A    /* deprecated */
	// #define CL_DEVICE_QUEUE_ON_HOST_PROPERTIES              0x102A
	// #define CL_DEVICE_NAME                                  0x102B
	// #define CL_DEVICE_VENDOR                                0x102C
	// #define CL_DRIVER_VERSION                               0x102D
	// #define CL_DEVICE_PROFILE                               0x102E


	CHECK_ERROR(err);

	printf("- CL_DEVICE_TYPE             :");
	if (device_type & CL_DEVICE_TYPE_CPU) printf(" CL_DEVICE_TYPE_CPU");  //각각 맞는 Type에 맞게 출력해준다.
	if (device_type & CL_DEVICE_TYPE_GPU) printf(" CL_DEVICE_TYPE_GPU");
	if (device_type & CL_DEVICE_TYPE_ACCELERATOR) printf(" CL_DEVICE_TYPE_ACCELERATOR");
	if (device_type & CL_DEVICE_TYPE_DEFAULT) printf(" CL_DEVICE_TYPE_DEFAULT");
	if (device_type & CL_DEVICE_TYPE_CUSTOM) printf(" CL_DEVICE_TYPE_CUSTOM");
	printf("\n");

	err = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 1024, str, NULL);  // 이건 Device Name을 출력해주는 함수

	CHECK_ERROR(err);

	printf("- CL_DEVICE_NAME        : %s\n", str);

	err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL); // 워크-그룹 하나의 워크-아이템 개수 제한 개수를 알아보는 것이다.

	CHECK_ERROR(err);

	printf("- CL_DEVICE_MAX_WORK_GROUP_SIZE  : %lu\n", max_work_group_size);

	err = clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL); // 글로벌 메모리 크기

	CHECK_ERROR(err);

	printf("- CL_DEVICE_GLOBLA_MEM_SIZE  : %lu\n", global_mem_size);

	err = clGetDeviceInfo(devices[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL); //로컬 메모리 크기

	CHECK_ERROR(err);

	printf("- CL_DEVICE_LOCAL_MEM_SIZE  : %lu\n", local_mem_size);

	err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_mem_alloc_size, NULL); // 메모리 오브젝트 크기 제한

	CHECK_ERROR(err);

	printf("- CL_DEVICE_MAX_MEM_ALLOC_SIZE  : %lu\n", local_mem_size);
}
free(devices);
}
free(platforms);
return 0;

}

