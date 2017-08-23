#### 파일 설명
  - main.c -> 기본 코드
  - main_analysis.c -> 분석코드
  - prog -> 실행 파일
  - .err .out -> 결과 파일, 오류 파일
  - kernel 파일 -> kernel.cl 파일에서 커널 소스 코드를 불러 온다. 이를 통해 프로그램 오브젝트를 만든다.
  
####  실습에 관한 내용

- 플랫폼을 아무거나 1개 얻어 온다
- 컨텍스트 만든다.
- 커맨드-큐를 만든다
- kernel.cl 파일에서 커널 소스 코드를 불러 온다
- 프로그램 오브젝트 만든다
- 프로그램 빌드
- 커널 오브젝트를 만든다. -> 만든 커널의 함수 이름은 my_kernel
- 지금까지 만들었던 모든 객체를 release -> 커널 오브젝트, 프로그램 오브젝트, 커맨드-큐, 컨텍스트


[스크린샷](https://www.evernote.com/l/AuH_9CSGZXVOnpun1ult5aiNJFWFYMZvrVc)
