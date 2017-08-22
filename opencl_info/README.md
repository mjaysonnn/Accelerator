#####  cl.h 헤더파일을 참고하기 위해 가져왔다. 
  - 헤더파일들을 공부하고 참조하기 위해 가져온 OpenCL 헤더파일

#### Compile
  - gcc -o opencl_info opencl_info.c -lOpenCL

#### Thorq에 queue 넣기
  - thorq --add --mode single --device gpu/1080 ./opencl_info
  - GPU1080에 돌리겠다는 뜻

#### 결과값 확인 법
  - thorq --stat 720659 (Queue Number이 주어진다)

##### stdout, stderr 이란?
  - 결과값을 출력해주는 파일, 오류가 뭔지 보여주는 파일

#### GPU1080 대신 GPU-7970으로 돌린 결과도 있어서 총 4개의 파일이 존재





