__kernel void reduction(__global int *g_num,
                        __global int *g_sum,
                        __local int *l_sum,
                        int TotalNum) {
  int i = get_global_id(0);
  int l_i = get_local_id(0);

  l_sum[l_i] = (i < TotalNum) ? g_num[i] : 0;  // global memory를 local memory 다 복사해준다.
  barrier(CLK_LOCAL_MEM_FENCE);  // 모든 work_item이 실행될때까지 기다린다.

  for (int p = get_local_size(0) / 2; p>=1 ; p=p>>1)
    if (l_i < p) l_sum[l_i] += l_sum[l_i + p];
    barrier(CLK_LOCAL_MEM_FENCE);  //reduction 1 이 다 실행될때까지 기다린다. 그 다음 reduction2


  if (l_i == 0) {
    g_sum[get_group_id(0)] = l_sum[0];// barrier를 거쳐야하므로 위의 for문이 끝난 이후이다.
  }
}