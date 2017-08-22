#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

int main() {
  // TODO
cl_uint num_platforms; // number of platform 
cl_platform_id *platforms; // platforms id
cl_uint num_devices; // number of device
cl_device_id *devices; // id of devices
char str[1024]; //strings to use
cl_device_type device_type; //type of device
size_t max_work_group_size; //work group size
cl_ulong global_mem_size; //global memory size
cl_ulong local_mem_size; //local memory size
cl_ulong max_mem_alloc_size; //maxmimum memory allocation size
cl_uint p, d; 
cl_uint err; //error to check if it's going right

err = clGetPlatformIDs(0, NULL, &num_platforms); // 플랫폼 개수 확인   num_entries, platform_id, num_of_platforms
CHECK_ERROR(err);

platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
err = clGetPlatformIDs(num_platforms, platforms, NULL);
CHECK_ERROR(err);

printf("Number of platforms: %u\n\n", num_platforms);
for( p =0; p < num_platforms; p++) {
	printf("platform: %u\n", p);

	err = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 1024, str, NULL);
	CHECK_ERROR(err);
	printf(" - CL_PLATFORM_NAME : %s\n", str);

	err = clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, 1024, str, NULL);
        CHECK_ERROR(err);
        printf(" - CL_VENDOR_NAME : %s\n\n", str);

	err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	CHECK_ERROR(err);
        printf("Number of devices : %u\n\n", num_devices);

	devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
	err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	CHECK_ERROR(err);

	for(d=0; d < num_devices; d++){
	printf("device: %u\n", d);
	
	err = clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
	CHECK_ERROR(err);
	printf("- CL_DEVICE_TYPE             :");
	if (device_type & CL_DEVICE_TYPE_CPU) printf(" CL_DEVICE_TYPE_CPU");
	if (device_type & CL_DEVICE_TYPE_GPU) printf(" CL_DEVICE_TYPE_GPU");
	if (device_type & CL_DEVICE_TYPE_ACCELERATOR) printf(" CL_DEVICE_TYPE_ACCELERATOR");
	if (device_type & CL_DEVICE_TYPE_DEFAULT) printf(" CL_DEVICE_TYPE_DEFAULT");
	if (device_type & CL_DEVICE_TYPE_CUSTOM) printf(" CL_DEVICE_TYPE_CUSTOM");
	printf("\n");
	
	err = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 1024, str, NULL);
	CHECK_ERROR(err);
	printf("- CL_DEVICE_NAME		: %s\n", str);
	
	err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
	CHECK_ERROR(err);
        printf("- CL_DEVICE_MAX_WORK_GROUP_SIZE  : %lu\n", max_work_group_size);

	err = clGetDeviceInfo(devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
        CHECK_ERROR(err);
        printf("- CL_DEVICE_GLOBLA_MEM_SIZE  : %lu\n", global_mem_size);

	err = clGetDeviceInfo(devices[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
        CHECK_ERROR(err);
        printf("- CL_DEVICE_LOCAL_MEM_SIZE  : %lu\n", local_mem_size);

	err = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_mem_alloc_size, NULL);
        CHECK_ERROR(err);
        printf("- CL_DEVICE_MAX_MEM_ALLOC_SIZE  : %lu\n", max_mem_alloc_size);
	}
	free(devices);
	}
	free(platforms);
	return 0;

}
