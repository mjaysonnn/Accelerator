__kernel void vec_add(__global int *A,
                      __global int *B,
                      __global int *C)
{
  int i = get_global_id(0);  //get_global_id란 뭘까? Returns the unique global work-item ID value for dimension identified by dimindx.
  C[i] = A[i] + B[i];
}
