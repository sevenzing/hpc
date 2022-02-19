__kernel void reg_mul_reg (
    const int N,
    __global const float* A,
    __global const float* B,
    __global float* C
) {
    // Get global indices of work-item
    int i = get_global_id(1);
    int j = get_global_id(0);
    // Check if we are within valid area of matrix C
    if( i < N && j < N ) {
        // Compute single element of C
        float s = 0;
        for (int k = 0; k < N; ++k)
            s += A[i * N + k] * B[k * N + j];
        // Write result into C matrix
        C[i * N + j] = s;
    }
}