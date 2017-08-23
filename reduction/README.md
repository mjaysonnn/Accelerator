#### 파일 설명

> atomic.stderr, stdout	- atomic operation을 했을 때 평균을 구하는 시간이 얼마나 걸리는지 보여준다. vs Sequential Code <br />
> kernel_atomic.cl - atomic 연산을 이용한 kernel , OpenCL C 언어로 된 코드
> kernel_local.cl	- local-memory 와 barrier를 이용한 kerenl, OpenCl C언어로 된 코드 
> local_barrier.stderr,stdout	- local_memory barrier 사용할때 보여주는 결과값
> reduction_atomic.c	- 호스트 프로그램 (atomic
> reduction_local_atomic.c - 호스트 프로그램(local-memory, barrier)
