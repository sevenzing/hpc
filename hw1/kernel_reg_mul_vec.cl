 
__kernel void reg_mul_vec(const int N, __global const float *mat, __global const float *vec, __global float *out) {
    int i = get_global_id(0);
    float val = 0;

    for (int j = 0; j < N; j++) {
        val += mat[i * N + j] * vec[j];
    }
    out[i] = val;
}
