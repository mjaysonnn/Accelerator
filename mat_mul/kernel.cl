__kernel void mat_mul(__global float *A, __global float *B, __global float *C, int ROW_A, int COL_A, int COL_B)
{
	int i = get_global_id(1);
	int j = get_global_id(0);
	int k;
	float sum = 0.0f;
	if (i < ROW_A && j < COL_B)
	{
		for(k=0; k< COL_A; k++)
		{
			sum += A[i * COL_A + k]*B[k*COL_B +j];
		}
		C[i*COL_B + j ] = sum;

	}
}

/*
 해당 인자는 6개이다
 병렬으로 처리하기 때문에 3중 for문이 아니다.
 global_size에 있는 2번째 인자값인 행을 불러오고
 1번째 인자값인 열을 불러온다.
 sum을 지정하고

 행 과 열이 전체 행렬 크기의 행과 열을 넘지 않는 범위에서

 elemnt-wise 가 아닌 column * row 끼리 곱해주고 다 더한 다음에 바로 C행렬에 저장을 해준다

 C는 row로 채워진다. 그게 더 기억하기 쉬울 것이다.

*/
