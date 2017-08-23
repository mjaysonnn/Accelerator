#### [이론](https://github.com/mjaysonnn/Accelerator/wiki/%EB%A1%9C%EC%BB%AC-%EB%A9%94%EB%AA%A8%EB%A6%AC%EC%99%80-%EC%9B%8C%ED%81%AC-%EA%B7%B8%EB%A3%B9%EA%B0%84%EC%9D%98-%EB%8F%99%EA%B8%B0%ED%99%94)

#### 16,777,216개 정수들의 평균을 구하는 프로그램 작성하기 실습

#### 파일 설명

> atomic.stderr, stdout	- atomic operation을 했을 때 평균을 구하는 시간이 얼마나 걸리는지 보여준다. vs Sequential Code <br />
> kernel_atomic.cl - atomic 연산을 이용한 kernel , OpenCL C 프로그램<br />
> kernel_local.cl	- local-memory 와 barrier를 이용한 kerenl, OpenCl C 프로그램<br /> 
> local_barrier.stderr,stdout	- local_memory barrier 사용할때 보여주는 프로그램<br />
> reduction_atomic.c	- 호스트 프로그램 (atomic)<br />
> reduction_local_atomic.c - 호스트 프로그램(local-memory, barrier)<br />
[결과값](https://www.evernote.com/l/AuGjiHLMJKVH37dcbeMPlI1S4rXd4WJsLS8)
