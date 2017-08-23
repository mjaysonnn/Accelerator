__kernel void reduction(__global int *g_num,
                        __global int *g_sum,
                        __local int *l_sum,
                        int TotalNum) {
  int i = get_global_id(0);
  if (i < TotalNum){
    atomic_add(g_sum, g_num[i]);
  }
}
