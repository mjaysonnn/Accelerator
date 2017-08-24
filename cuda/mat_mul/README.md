#### 파일 설명

> mat_mul_cuda.cu -> cuda를 이용해 matrix multiplication 한 프로그램

> kernel.cl , mat_mul_opencl.cu -> cu 파일은 호스트 프로그램에서 도 쓰이기 떄문에 이렇게도 가능 , 기존 OpenCl 행렬 곱셈 프로그램

> mat_mul_cublas.cu -> cublas를 이용한 행렬 곱셈 프로그램
(single_device), 제일 빠름

> mat_mul_cuda_multi.cu -> Multi_deve에서 cuda를 이용한 행렬 곱셈 프로그램

> mat_mul_seq.cu -> C 순차적인 코드 행렬 곱셈 프로그램

>mat_mul.cu -> main 함수, 옵션에 따라 여러가지 sequential , opencl, cublas 를 사용하게 함, latency도 측정해준다.

##### thorq 에 넣는 법
~~~
thorq --add --mode single --device gpu/1080 --name cublas ./mat_mul 4
~~~
##### 옵션
0. sequence
1. opencl
2. cuda
3. cuda with multiple devices
4. cublas

[결과](https://www.evernote.com/l/AuEbT6ZoFHtN75ISt0H6u3A1tdEznbl1nAw)
