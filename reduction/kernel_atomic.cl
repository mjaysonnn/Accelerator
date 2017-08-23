__kernel void reduction(__global int *g_num,
                        __global int *g_sum,
                        __local int *l_sum,
                        int TotalNum) {
  int i = get_global_id(0);       // global size이다.
  if (i < TotalNum){              
    atomic_add(g_sum, g_num[i]);  // g_sum은 host program에서 초기화 해준다. buf_sum이라는 걸 ClEnqueuWriteBuffer를 통해 초기화 해준걸 알수있다.

  }
}

; /*
; buf_sum=clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &err); 

; int sum = 0;  // sum은 0으로 초기화되어있다.
; err = clEnqueueWriteBuffer(queue, buf_sum, CL_FALSE, 0, sizeof(int), &sum, 0, NULL, NULL); // sum 이라는 데이터를 buf_sum에 또 다시 써준다.. 이렇게 buf_sum을 0으로 초기화 된다.
; */