double f(double x) { return (3*x*x + 2*x +1);}

__kernel void integral( int N,
                        __global double *g_sum,
                        __local  double *l_sum) {
  int i = get_global_id(0); // N임                                            20이라고 하고 Work-Group Size는 6        24
  int l_i = get_local_id(0); // Local_size의 하나의 인덱스임                                                   7         1

  double dx = (1000.0 / (double)N);
  l_sum[l_i] = (i < N) ? f(i *dx)*dx:0;  //             l_sum[1]=g_num[1]  이제 1에 해당하는 모든 워크 아이템들이 다 실행될때까지 기다려야겠네 결국 여기서 모든 워크-아이템이 이걸 다 하고 지나갈듯
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int p = get_local_size(0) / 2; p >= 1; p = p >> 1) {    // 32 16 4  2  1  이렇게 되는구나
    if (l_i < p) l_sum[l_i] += l_sum[l_i + p];                 // 워크 그룹 안의 모든게 연산이 되어야 한다. 그래서 1이 될때까지 계속 합쳐지구나
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (l_i == 0) {                                              // l_i가 0이 되면
    g_sum[get_group_id(0)] = l_sum[0];                         // l_sum[0]이 워크 그룹에서의 총 합이기 떄문에 이걸 모두 더해준다.
  }                                                            // get_group_id -> 워크-그룹의 인덱스이다.
}
