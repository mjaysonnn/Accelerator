#### 파일 설명
> Makefile -> -lOpenCL 컴파일 옵션이 들어있다. <br />
> bumfuncs.c, bumfuncs.h -> 비트맵 파일을 읽고 쓰는 코드가 포함되어 있다. <br />
> image_rotation.c -> 호스트 프로그램 OpenCL API 함수를 사용해 디바이스를 관리한다.<br />
> image_rotation_analysis.c -> 분석코드<br />
> input.bmp -> 회전을 시킬 Input Image <br />
> kernel.cl -> 커널, OpenCL C언어로 작성, 디바이스에서 실행되는 기본 단위 <br />
> result.bmp -> 결과 그림 회전되어있음<br />
> result1.bmp -> kernel의 인덱스 공간을 600*400 대신 400*600으로 잡았을 경우 ( 커널에서 수정해봤다.) -스샷 참고<br />
> task.* -> Thorq에 돌렸을때 나타는 결과값 ,오류값<br />


[결과 스크린 샷](https://www.evernote.com/l/AuHpdQhKrxNPcrNpYUep3Yc1C1hCWfMKVmU)
