#### 파일
kernel.cl	  -> OpenCL 커널 함수 , 커널 함수를 가지고 프로그램 오브젝트를 만든다.
main.c	-> 기본 메인 함수
main_analysis.c	-> 기본 메인 함수 분석해봄
vector.stderr	->  오류 출력해주는 값
vector.stdout ->  결과 출력해주는 값

#### Compile & Thorq Add 

gcc -o main main.c -lOpenCL


thorq --add --mode single --device gpu/1080 --name vector ./main


[결과값링크](https://www.evernote.com/l/AuHiVsWebqhOvLcbQ071b1recopw0O2Djvo)
