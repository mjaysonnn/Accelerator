__kernel void img_rotate(__global float * dest_data,
			                   __global float* src_data,
			                   int W, int H,
			                   float sinTheta, float cosTheta)
{
  int dest_x = get_global_id(0); // global size의 첫번째 값
  int dest_y = get_global_id(1); // 출력 이미지는 데이터 디펜던스가 없음
  if (dest_x >= W || dest_y >= H) return;
  float x0 = W / 2.0f;  //좌표로 보면 점의 한 가운데 x축
  float y0 = H / 2.0f;  //좌표로 보면 점의 한 가운데 y축
  float xOff = dest_x - x0;                                //(x2,y2)는 (x0, y0)를 기준으로 (x1,y1)를 theta 만큼 회전함 점
  float yOff = dest_y - y0;
  int src_x = (int)(xOff*cosTheta + yOff*sinTheta + x0);  // (src_x,srx_y)는 (x0, y0)를 기준으로 (xOff,yOff)를 theta 만큼 움직인 점
  int src_y = (int)(yOff*cosTheta - xOff*sinTheta + y0);
  if((src_x >= 0) && (src_x < W ) && (src_y >= 0) && (src_y < H)) {
  dest_data[dest_y*W+dest_x] = src_data[src_y*W + src_x]; //병렬처리를 하기 떄문에 W까지 한꺼번에 한다. x1과 y1은 제일 끝 부분으로 처리한다.
 }
 else {
  dest_data[dest_y*W+dest_x] = 0.0f;
 }
}
