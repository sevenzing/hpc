__kernel void crsMulVec(
    const int N, 
    __global const float *values, 
    __global const int *col, 
    __global const int *row_index, 
    __global const float *v, 
    __global float *out)
{
    int i = get_global_id(0);
    float s = 0;
    int j1 = row_index[i];
    int j2 = row_index[i + 1];
    for (int j = j1; j < j2; j++){
        s += v[col[j]] * values[j];
    };
    out[i] = s;
}
