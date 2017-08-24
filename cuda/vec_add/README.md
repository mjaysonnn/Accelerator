### 두 벡터를 더하는 CUDA 프로그램

> thread_block 32 로 했을 때 커널 실행시간(메모리 복사 , 커널 실행, 메모리 다시 메인 메모리로)  알려준다.

> 커널 실행은 non_blocking이라서 Background에서 돌아감

> thread_block 64 로 했을 때 커널 실행시간을 알려준다.

### 컴파을 하는 방법

~~~bash
nvcc vec_add.cu -o vec_add
~~~

[결과](https://www.evernote.com/l/AuGizDjDZfNKQrajYAXRvo1qTRubCfALvIQ)
