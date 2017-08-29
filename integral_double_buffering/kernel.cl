double f(double x) { return (3*x*x + 2*x +1);}

__kernel void integral( int N, int base, int offset, __global double *g_sum, __local  double *l_sum)  // g_sum 이건 인자로 없다는게 함정이네 ,
{
  int i = get_global_id(0); // N임                                            20이라고 하고 Work-Group Size는 6        24
  int l_i = get_local_id(0); // Local_size의 하나의 인덱스임                                                   7         1

  double dx = (1000.0 / (double) N);

  l_sum[l_i] = (i < offset) ? f((i + base) *dx ) * dx : 0;  //             l_sum[1]=g_num[1]  이제 1에 해당하는 모든 워크 아이템들이 다 실행될때까지 기다려야겠네 결국 여기서 모든 워크-아이템이 이걸 다 하고 지나갈듯
  barrier(CLK_LOCAL_MEM_FENCE);                             // 초기화 해줄때 값도 거의 지정해주네 . reduction이랑 많이 비슷하네

  for (int p = get_local_size(0) / 2; p >= 1; p = p >> 1)
  {
    if (l_i < p) l_sum[l_i] += l_sum[l_i + p];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (l_i == 0) {
    g_sum[get_group_id(0)] = l_sum[0];   //buf_partial_sum이 global memory의 절반이라고 생각하면 된다.
  }
}