#### 파일 설명
 
 - .h, .hpp -> 공부하고 참조하기 위해 가져온 OpenCL 헤더파일
 - example.out -> 실습 정답
 - stderr, stdout -> 천둥에서 돌려서 나온 결과값 , 오류 파일
 - GPU1080 대신 GPU-7970으로 돌린 결과도 있어서 총 4개의 파일이 존재
 - .c 파일 -> 기본 코드랑 분석한 코드 (_analysis ) , C99기반의 언어이다.
#### Compile
  - gcc -o opencl_info opencl_info.c -lOpenCL

#### Thorq에 queue 넣기
  - thorq --add --mode single --device gpu/1080 ./opencl_info
  - GPU1080에 돌리겠다는 뜻

#### 결과값 확인 법
  - thorq --stat 720659 (Queue Number이 주어진다)
  
[스샷](https://www.evernote.com/l/AuGYTOBhdfhH64jU0dfuFwS93nWgrWrbIrE)






