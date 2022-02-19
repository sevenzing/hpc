#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "tools.c"
#include "regMatrix.c"
#include "crsMatrix.c"


#define CL_TARGET_OPENCL_VERSION 300

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


#define ZERO_IN_CRS 0
#define ZERO_IN_REG 0.0001
#define MAX_ITEMS 10
#define MAX_SOURCE_SIZE 2048

#define foreach(item, array) \
    for(int keep = 1, \
            count = 0,\
            size = sizeof (array) / sizeof *(array); \
        keep && count != size; \
        keep = !keep, count++) \
      for(item = (array) + count; keep; keep = !keep)


#define p(a) printf("here%d\n", (a))


#define DIE(ret) \
do { \
    int _ret = ret; \
    if (_ret != CL_SUCCESS) { \
        fprintf(stderr, "Die at line %d. Exit code %d\n", __LINE__, _ret); \
        exit(1); \
    } \
} while (0)

#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),  (mode)))==NULL
#endif


void print_platform(cl_platform_id id) {
    cl_device_id *devices = malloc(MAX_ITEMS * sizeof(cl_device_id));
    cl_uint num_devices;

    clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, MAX_ITEMS, devices, &num_devices);
    
    for (cl_uint i = 0; i < num_devices; i++) {
        char* name = malloc(255);
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 255, name, NULL);
        printf("GPU name: %s\n", name);
        free(name);
        
        cl_ulong res;
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(res), &res, NULL);
        printf("Clock frequency: %ld\n", res);
        
        clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(res), &res, NULL);
        printf("Number of shared cores: %ld\n", res);
        
        clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(res), &res, NULL);
        printf("GRAM: %ld\n", res);
    }
    free(devices);
}

void printVector(vector_of(float) *v) {
    vector_iter(v, _, float item, {
        printfloat(item);
    });
}


// Task 8
regMatrix regFromCrs(crsMatrix *cm){
    regMatrix rm = newReg(cm->n);
    vector_iter(&cm->row_index, row, int columns_index_start, {
        if ((int)row == cm->n) {
            break;
        }
        int columnds_index_end = cm->row_index[row + 1];
        for (int column_index = columns_index_start; column_index < columnds_index_end; column_index++){
            int column = cm->col[column_index];
            rm.values[row][column] = cm->value[column_index];
        }
    });
    return rm;
}
    
// Task 9
void regMultVector(regMatrix *rm, vector_of(float) *x, vector_of(float) *r, float* result_time){
    assert(rm->n == (int)vector_len(x));

    *result_time = omp_get_wtime();
    for (int row = 0; row < rm->n; row++){
        float s = 0; 
        vector_iter(x, column, float value, {
            s += value * rm->values[row][column];
        });
        vector_push(r, s);
    }
    *result_time = omp_get_wtime() - *result_time;
}

// Task 10

void crsMultVector(crsMatrix *cm, vector_of(float) *x, vector_of(float) *y, float *result_time){
    assert(cm->n == (int)vector_len(x));

    *result_time = omp_get_wtime();
    for (int row = 0; row < cm->n; row++){
        float s = 0;
        int j1 = cm->row_index[row];
        int j2 = cm->row_index[row + 1];
        for (int j = j1; j < j2; j++){
            s += (*x)[cm->col[j]] * cm->value[j];
        };
        vector_push(y, s);
    }
    *result_time = omp_get_wtime() - *result_time; 
}

// Task 11.
float compareVectors(vector_of(float) *vec1, vector_of(float) *vec2, int n){
    float diff = 0;
    for (int i = 0; i < n; i++){
        float current_diff = fabs((*vec1)[i] - (*vec2)[i]); 
        if (diff < current_diff){
            diff = current_diff;
        }
    }
    return diff;
}

// Task 13.
void regMultVectorOmp(regMatrix *rm, vector_of(float) *x, vector_of(float) *r, float* result_time){
    assert(rm->n == (int)vector_len(x));
    for (int i = 0; i < rm->n; i++)
        vector_push(r, 0);
    *result_time = omp_get_wtime();
    #pragma omp parallel for
    for (int row = 0; row < rm->n; row++){
        float s = 0; 
        for (int column = 0; column < rm->n; column++) {
            s = s + ((*x)[column]) * (rm->values[row][column]);
        };
        (*r)[row] = s;
    }
    *result_time = omp_get_wtime() - *result_time; 
}

// Task14

void printPoint(float x, float y){
    printf("%f\t%f\n",x ,y);
}

void testRegMulVector(){
    for (int n = 500; n <= 5000; n += 500) {
        crsMatrix cm = newCrsSpecial(1, n, 3);
        regMatrix rm = regFromCrs(&cm);

        vector_decl(float, x);
        for (int i = 0; i < n; i++)
            vector_push(&x, n / 2 - i);
        
        vector_decl(float, r);
        float time;
        
        regMultVector(&rm, &x, &r, &time);
        printPoint(n, time);
    }
}

void testRegMulVectorOmp(){
    for (int n = 500; n <= 5000; n += 500) {
        crsMatrix cm = newCrsSpecial(1, n, 3);
        regMatrix rm = regFromCrs(&cm);

        vector_decl(float, x);
        for (int i = 0; i < n; i++)
            vector_push(&x, n / 2 - i);
        
        vector_decl(float, r);
        float time;
        
        regMultVectorOmp(&rm, &x, &r, &time);
        printPoint(n, time);
    }
}

// Task 15.
void crsMultVectorOmp(crsMatrix *cm, vector_of(float) *x, vector_of(float) *y, float *result_time){
    assert(cm->n == (int)vector_len(x));
    for (int i = 0; i < cm->n; i++)
        vector_push(y, 0);

    *result_time = omp_get_wtime();
    #pragma omp parallel for
    for (int row = 0; row < cm->n; row++){
        float s = 0;
        int j1 = cm->row_index[row];
        int j2 = cm->row_index[row + 1];
        for (int j = j1; j < j2; j++){
            s += (*x)[cm->col[j]] * cm->value[j];
        };
        (*y)[row] = s;
    }
    *result_time = omp_get_wtime() - *result_time; 
}

// Task 16.
void testCrsMulVector(){
    for (int n = 5000; n <= 40000; n += 5000) {
        crsMatrix cm = newCrsSpecial(1, n, 3);

        vector_decl(float, x);
        for (int i = 0; i < n; i++)
            vector_push(&x, n / 2 - i);
        
        vector_decl(float, r);
        float time;
        
        crsMultVector(&cm, &x, &r, &time);
        printPoint(n, time);
    }
}

void testCrsMulVectorOmp(){
    for (int n = 5000; n <= 40000; n += 5000) {
        crsMatrix cm = newCrsSpecial(1, n, 3);

        vector_decl(float, x);
        for (int i = 0; i < n; i++)
            vector_push(&x, n / 2 - i);
        
        vector_decl(float, r);
        float time;
        
        crsMultVectorOmp(&cm, &x, &r, &time);
        printPoint(n, time);
    }
}

// Task 17.
regMatrix regMulReg(regMatrix a, regMatrix b, float *result_time) {
    *result_time = omp_get_wtime();
    regMatrix out = newReg(a.n);
    for (int i = 0; i < a.n; i++) {
        for (int j = 0; j < a.n; j++) {
            float value = 0;
            for (int k = 0; k < a.n; k++) {
                value += a.values[i][k] * b.values[k][j];
            }
        out.values[i][j] = value;
        }
    }
    *result_time = omp_get_wtime() - *result_time; 
    return out;
}

// Task 18.

crsMatrix transposeCrs(crsMatrix imtx) {
  int i, j;


  crsMatrix omtx = newCrs(imtx.n, imtx.nz);

  memset(omtx.row_index, 0, (imtx.n + 1) * sizeof(int));
  for (i = 0; i < imtx.nz; i++)
    omtx.row_index[imtx.col[i] + 1]++;
  
  int S = 0;
  for (i = 1; i <= imtx.n; i++) 
  {
    int tmp = omtx.row_index[i];
    omtx.row_index[i] = S;
    S = S + tmp;
  }

  for (i = 0; i < imtx.n; i++) 
  {
    int j1 = imtx.row_index[i];
    int j2 = imtx.row_index[i+1];
    int Col = i; 
    for (j = j1; j < j2; j++) 
    {
      float V = imtx.value[j]; 
      int RIndex = imtx.col[j]; 
      int IIndex = omtx.row_index[RIndex + 1];
      omtx.value[IIndex] = V;
      omtx.col[IIndex] = Col;
      omtx.row_index[RIndex + 1]++;
    }
  }

  return omtx;
}

crsMatrix crsMulCrs(crsMatrix a, crsMatrix _b, float *result_time)
{
  vector_decl(float, values);
  vector_decl(int, columns);
  vector_decl(int, row_index);

  int N = a.n;
  int NZ = 0;

  *result_time = omp_get_wtime();
  crsMatrix b = transposeCrs(_b);
  vector_push(&row_index, 0);
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      // Умножаем строку i матрицы A и столбец j матрицы B
      float sum = 0;
      int ks = a.row_index[i]; int ls = b.row_index[j]; 
      int kf = a.row_index[i + 1] - 1; int lf = b.row_index[j + 1] - 1;
      while ((ks <= kf) && (ls <= lf)){
        if (a.col[ks] < b.col[ls])
          ks++;
        else 
          if (a.col[ks] > b.col[ls])
            ls++;
          else 
          {
            sum += a.value[ks] * b.value[ls];
            ks++;
            ls++;
          }
      }
      if (fabs(sum) > ZERO_IN_CRS)
      {
        vector_push(&values, sum);
        vector_push(&columns, j);
        NZ++;
      }
    }
    vector_push(&row_index, NZ);
  }

  crsMatrix c = newCrs(N, vector_len(&columns));

  for (unsigned int j = 0; j < vector_len(&columns); j++){
    c.value[j] = values[j];
    c.col[j] = columns[j];
  }
  for (int i = 0; i <= N; i++)
    c.row_index[i] = row_index[i];
  *result_time = omp_get_wtime() - *result_time; 
  return c;
}


// Task 19.

int checkValue(regMatrix rm, int i, int j, float value){
    if (fabs(rm.values[i][j] - value) > ZERO_IN_REG){
        printf("value at [%d][%d] does not equal to %f, but %f\n", i, j, value, rm.values[i][j]);
        return 1;
    }
    return 0;
}

int checkEq(crsMatrix a, regMatrix b){
    vector_iter(&a.row_index, row, int columns_index_start, {
        if ((int)row == a.n) {
            break;
        }
        int columnds_index_end = a.row_index[row + 1];
        int j = 0;
        for (int column_index = columns_index_start; column_index < columnds_index_end; column_index++){
            int column = a.col[column_index];
            while (j < column){
                if (checkValue(b, row, j, 0) == 1) return 1;              
                j++;
            }
            if (checkValue(b, row, j, a.value[column_index]) == 1) return 1;
            j++;
        }
        while (j < a.n) {
            if (checkValue(b, row, j, 0) == 1) return 1;
            j++;
        }
    });
    return 0;
}

// Task 20.
regMatrix regMulRegOmp(regMatrix a, regMatrix b, float *result_time) {
    *result_time = omp_get_wtime();
    regMatrix out = newReg(a.n);
    #pragma omp parallel for
    for (int i = 0; i < a.n; i++) {
        for (int j = 0; j < a.n; j++) {
            float value = 0;
            for (int k = 0; k < a.n; k++) {
                value += a.values[i][k] * b.values[k][j];
            }
        out.values[i][j] = value;
        }
    }
    *result_time = omp_get_wtime() - *result_time; 
    return out;
}


// Task 21.
void testRegMulReg(){
    for (int n = 100; n <= 1000; n += 100) {
        crsMatrix cm_1 = newCrsSpecial(1, n, 2);
        crsMatrix cm_2 = newCrsSpecial(2, n, 2);

        regMatrix rm_1 = regFromCrs(&cm_1);
        regMatrix rm_2 = regFromCrs(&cm_2);
    
        float time = 0;
        regMatrix rm_o = regMulReg(rm_1, rm_2, &time);
        printPoint(n, time);
    }
}

void testRegMulRegOmp(){
    for (int n = 100; n <= 1000; n += 100) {
        crsMatrix cm_1 = newCrsSpecial(1, n, 2);
        crsMatrix cm_2 = newCrsSpecial(2, n, 2);

        regMatrix rm_1 = regFromCrs(&cm_1);
        regMatrix rm_2 = regFromCrs(&cm_2);
    
        float time = 0;
        regMatrix rm_o = regMulRegOmp(rm_1, rm_2, &time);
        printPoint(n, time);
    }
}

// Task 22.

crsMatrix crsMulCrsOmp(crsMatrix a, crsMatrix _b, float *result_time)
{
  vector_decl(float, values);
  vector_decl(int, columns);
  vector_decl(int, row_index);

  int N = a.n;
  int NZ = 0;

  *result_time = omp_get_wtime();
  crsMatrix b = transposeCrs(_b);
  vector_push(&row_index, 0);
  
  #pragma omp parallel for shared(NZ)
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      // Умножаем строку i матрицы A и столбец j матрицы B
      float sum = 0;
      int ks = a.row_index[i]; int ls = b.row_index[j]; 
      int kf = a.row_index[i + 1] - 1; int lf = b.row_index[j + 1] - 1;
      while ((ks <= kf) && (ls <= lf)){
        if (a.col[ks] < b.col[ls])
          ks++;
        else 
          if (a.col[ks] > b.col[ls])
            ls++;
          else 
          {
            #pragma omp atomic
            sum += a.value[ks] * b.value[ls];
            ks++;
            ls++;
          }
      }
      #pragma omp critical
      if (fabs(sum) > ZERO_IN_CRS){
        vector_push(&values, sum);
        vector_push(&columns, j);
        NZ++;
      }
    }
    #pragma omp critical
    vector_push(&row_index, NZ);
  }

  crsMatrix c = newCrs(N, vector_len(&columns));

  for (unsigned int j = 0; j < vector_len(&columns); j++){
    c.value[j] = values[j];
    c.col[j] = columns[j];
  }
  for (int i = 0; i <= N; i++)
    c.row_index[i] = row_index[i];
  *result_time = omp_get_wtime() - *result_time; 
  return c;
}


void testCrsMulCrs(){
    for (int n = 500; n <= 5000; n += 500) {
        crsMatrix cm_1 = newCrsSpecial(1, n, 2);
        crsMatrix cm_2 = newCrsSpecial(2, n, 2);
    
        float time = 0;
        crsMulCrs(cm_1, cm_2, &time);
        printPoint(n, time);
    }
}

void testCrsMulCrsOmp(){
    for (int n = 500; n <= 5000; n += 500) {
        crsMatrix cm_1 = newCrsSpecial(1, n, 2);
        crsMatrix cm_2 = newCrsSpecial(2, n, 2);
    
        float time = 0;
        crsMulCrsOmp(cm_1, cm_2, &time);
        printPoint(n, time);
    }
}

// Task 24.
vector_of(float) regMulVecGpu(regMatrix m, vector_of(float) v, float *time){
    // Load the kernel source code into the array source_str
    FILE* fp;
    char* source_str;
    size_t source_size;

    fopen_s(&fp, "kernel_reg_mul_vec.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    DIE(ret);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1,
        &device_id, &ret_num_devices);
    DIE(ret);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    DIE(ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
    DIE(ret);

    // Create memory buffers on the device for each vector 
    cl_mem mat_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * m.n * m.n, NULL, &ret);
    DIE(ret);
    cl_mem vec_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * m.n, NULL, &ret);
    DIE(ret);
    cl_mem out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * m.n, NULL, &ret);
    DIE(ret);

    float** mat_values = get_flatten(m);

    ret = clEnqueueWriteBuffer(command_queue, mat_buf, CL_TRUE, 0,
        sizeof(float) * m.n * m.n, mat_values, 0, NULL, NULL);
    DIE(ret);
    ret = clEnqueueWriteBuffer(command_queue, vec_buf, CL_TRUE, 0,
        sizeof(float) * m.n, v, 0, NULL, NULL);
    DIE(ret);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, (const size_t*)&source_size, &ret);
    DIE(ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    DIE(ret);
    int valueSize;
    DIE(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &valueSize));
    char* buildlog = (char*)malloc(valueSize);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, valueSize, buildlog, NULL);
    // printf("Buildlog:   %s\n", buildlog);
    cl_kernel kernel = clCreateKernel(program, "reg_mul_vec", &ret);
    DIE(ret);    
    ret = clSetKernelArg(kernel, 0, sizeof(cl_uint), &m.n);
    DIE(ret);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mat_buf);
    DIE(ret);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&vec_buf);
    DIE(ret);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&out_buf);
    DIE(ret);
    size_t global_item_size = m.n;
    size_t local_item_size = 10;
    *time = omp_get_wtime();
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_item_size, &local_item_size, 0, NULL, NULL);
    DIE(ret);
    
    vector_decl(float, out);
    for (int i = 0; i < m.n; i++)
        vector_push(&out, 0);

    ret = clEnqueueReadBuffer(command_queue, out_buf, CL_TRUE, 0,
        m.n * sizeof(float), out, 0, NULL, NULL);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(mat_buf);
    ret = clReleaseMemObject(vec_buf);
    ret = clReleaseMemObject(out_buf);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    *time = omp_get_wtime() - *time; 
    return out;
}

// Task 26.
void testRegMulVecGpu(){
    for (int n = 5000; n <= 15000; n += 2500) {
        crsMatrix cm = newCrsSpecial(1, n, 3);
        regMatrix rm = regFromCrs(&cm);

        vector_decl(float, x);
        for (int i = 0; i < n; i++)
            vector_push(&x, n / 2 - i);
        
        vector_decl(float, r);
        float time;
        
        regMulVecGpu(rm, x, &time);
        printPoint(n, time);
    }
}


vector_of(float) crsMulVecGpu(crsMatrix m, vector_of(float) v, float *time){
    *time = omp_get_wtime();

    // Load the kernel source code into the array source_str
    FILE* fp;
    char* source_str;
    size_t source_size;

    fopen_s(&fp, "kernel_crs_mul_vec.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    DIE(ret);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1,
        &device_id, &ret_num_devices);
    DIE(ret);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    DIE(ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
    DIE(ret);

    // Create memory buffers on the device for each vector 
    cl_mem values_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * m.nz, NULL, &ret);
    DIE(ret);
    cl_mem col_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(int) * m.nz, NULL, &ret);
    DIE(ret);
    cl_mem row_index_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(int) * (m.n + 1), NULL, &ret);
    DIE(ret);
    cl_mem v_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * m.n, NULL, &ret);
    DIE(ret);
    cl_mem out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * m.n, NULL, &ret);
    DIE(ret);

    ret = clEnqueueWriteBuffer(command_queue, values_buf, CL_TRUE, 0,
        sizeof(float) * m.nz, m.value, 0, NULL, NULL);
    DIE(ret);
    ret = clEnqueueWriteBuffer(command_queue, col_buf, CL_TRUE, 0,
        sizeof(int) * m.nz, m.col, 0, NULL, NULL);
    DIE(ret);
    ret = clEnqueueWriteBuffer(command_queue, row_index_buf, CL_TRUE, 0,
        sizeof(int) * (m.n + 1), m.row_index, 0, NULL, NULL);
    DIE(ret);
    ret = clEnqueueWriteBuffer(command_queue, v_buf, CL_TRUE, 0,
        sizeof(float) * m.n, v, 0, NULL, NULL);
    DIE(ret);
    
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, (const size_t*)&source_size, &ret);
    DIE(ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    DIE(ret);
    int valueSize;
    DIE(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &valueSize));
    char* buildlog = (char*)malloc(valueSize);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, valueSize, buildlog, NULL);
    // printf("Buildlog:   %s\n", buildlog);
    cl_kernel kernel = clCreateKernel(program, "crsMulVec", &ret);
    DIE(ret);    
    ret = clSetKernelArg(kernel, 0, sizeof(cl_uint), &m.n);
    DIE(ret);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &values_buf);
    DIE(ret);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &col_buf);
    DIE(ret);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &row_index_buf);
    DIE(ret);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &v_buf);
    DIE(ret);
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &out_buf);
    DIE(ret);

    clFlush(command_queue);
    clFinish(command_queue);

    size_t global_item_size = m.n;
    size_t local_item_size = 10;
    
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
        &global_item_size, &local_item_size, 0, NULL, NULL);
    DIE(ret);
    
    vector_decl(float, out);
    for (int i = 0; i < m.n; i++)
        vector_push(&out, 0);

    ret = clEnqueueReadBuffer(command_queue, out_buf, CL_TRUE, 0,
        m.n * sizeof(float), out, 0, NULL, NULL);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(values_buf);
    ret = clReleaseMemObject(col_buf);
    ret = clReleaseMemObject(row_index_buf);
    ret = clReleaseMemObject(v_buf);
    ret = clReleaseMemObject(out_buf);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    *time = omp_get_wtime() - *time; 
    return out;
}

void testCrsMulVecGpu(){
    for (int n = 5000; n <= 40000; n += 5000) {
        crsMatrix cm = newCrsSpecial(1, n, 3);

        vector_decl(float, x);
        for (int i = 0; i < n; i++)
            vector_push(&x, n / 2 - i);
        
        vector_decl(float, r);
        float time;
        
        crsMulVecGpu(cm, x, &time);
        printPoint(n, time);
    }
}

// Task 30.

regMatrix regMulRegGpu(regMatrix a, regMatrix b, float *time){
    // Load the kernel source code into the array source_str
    FILE* fp;
    char* source_str;
    size_t source_size;

    fopen_s(&fp, "kernel_reg_mul_reg.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    DIE(ret);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1,
        &device_id, &ret_num_devices);
    DIE(ret);
    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    DIE(ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
    DIE(ret);

    // Create memory buffers on the device for each vector 
    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * a.n * a.n, NULL, &ret);
    DIE(ret);
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * b.n * b.n, NULL, &ret);
    DIE(ret);
    cl_mem c_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * a.n * b.n, NULL, &ret);
    DIE(ret);

    float** a_values = get_flatten(a);
    float** b_values = get_flatten(b);

    ret = clEnqueueWriteBuffer(command_queue, a_buf, CL_TRUE, 0,
        sizeof(float) * a.n * a.n, a_values, 0, NULL, NULL);
    DIE(ret);
    ret = clEnqueueWriteBuffer(command_queue, b_buf, CL_TRUE, 0,
        sizeof(float) * b.n * b.n, b_values, 0, NULL, NULL);
    DIE(ret);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, (const size_t*)&source_size, &ret);
    DIE(ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    DIE(ret);
    int valueSize;
    DIE(clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &valueSize));
    char* buildlog = (char*)malloc(valueSize);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, valueSize, buildlog, NULL);
    // printf("Buildlog:   %s\n", buildlog);
    cl_kernel kernel = clCreateKernel(program, "reg_mul_reg", &ret);
    DIE(ret);    
    ret = clSetKernelArg(kernel, 0, sizeof(cl_uint), &a.n);
    DIE(ret);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&a_buf);
    DIE(ret);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&b_buf);
    DIE(ret);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&c_buf);
    DIE(ret);
    size_t global_item_size[2] = {a.n, a.n};
    size_t local_item_size[2] = {10, 10};
    *time = omp_get_wtime();
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
        global_item_size, local_item_size, 0, NULL, NULL);
    DIE(ret);
    
    regMatrix c = newReg(a.n);
    float* c_values = (float*)malloc(a.n * a.n * sizeof(float));

    ret = clEnqueueReadBuffer(command_queue, c_buf, CL_TRUE, 0,
        a.n * a.n * sizeof(float), c_values, 0, NULL, NULL);

    for (int i = 0; i < a.n; i++){
        for (int j = 0; j < a.n; j++) {
            c.values[i][j] = c_values[i * a.n + j];
        }
    }


    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_buf);
    ret = clReleaseMemObject(b_buf);
    ret = clReleaseMemObject(c_buf);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    *time = omp_get_wtime() - *time; 
    return c;
}



int main(void) {
    int seed = 1488;
    FILE *saved_stdout = stdout;

    // Task 1-3 start
    printf("Lev Lymarenko \nProgrammin assignment #01\n\n== CPU INFO ==\n");
    system("lscpu | grep \"Model name\"");
    system("lscpu | grep \"MHz\"");
    system("lscpu | grep -G \"^CPU(s)\"");
    system("lsmem | grep \"Total online memory\"");

    printf("\n== GPU INFO ==\n");
    cl_platform_id *ids = malloc(MAX_ITEMS * sizeof(cl_platform_id));
    cl_uint total_platforms;

    clGetPlatformIDs(MAX_ITEMS, ids, &total_platforms);
    for (cl_uint i = 0; i < total_platforms; i++) {
        print_platform(ids[i]);
    }
    free(ids);
    // Task 1-3 end
    
    // Task 12
    printf("\nTask 12. Checking matrix multiplication\n");
    int sizes[] = {1e1, 1e2, 1e3};
    foreach(int *n, sizes) {
        float time;
        crsMatrix cm = newCrsSpecial(seed, *n, *n / 10);
        regMatrix rm = regFromCrs(&cm);

        vector_decl(float, x);
        for (int i = 0; i < *n; i++){
            vector_push(&x, *n / 2 - i);
        }
        vector_decl(float, crs_result);
        crsMultVector(&cm, &x, &crs_result, &time);

        vector_decl(float, reg_result);
        regMultVector(&rm, &x, &reg_result, &time);
        
        printf("For n = %d: ", *n);
        if (compareVectors(&crs_result, &reg_result, *n) == 0){
            printf("correct");
        } else{
            printf("incorrect");
        }
        printf("\n");
    }

    // Task 14.
    printf("\nRegular matrix-vector multiplication serials:\n");
    //stdout = fopen("points/reg-vec/1-cores", "w");
    testRegMulVector();
    //fclose(stdout);
    //stdout = saved_stdout;

    for (int cores = 2; cores <= 8; cores *= 2) {
        printf("\nRegular matrix-vector multiplication with OMP with %d core(s):\n", cores);
        omp_set_num_threads(cores);
        testRegMulVectorOmp();
    }
    
    // Task 16.
    printf("\nCrs matrix-vector multiplication serials:\n");
    testCrsMulVector();

    for (int cores = 2; cores <= 8; cores *= 2) {
        printf("\nCrs matrix-vector multiplication with OMP with %d core(s):\n", cores);
        omp_set_num_threads(cores);
        testCrsMulVectorOmp();
    }

    // Task 19.
    printf("\nCheck crs-crs and regular-regular matrix mult\n");
    for (int n = 10; n <= 1000; n *= 10){
        crsMatrix cm_1 = newCrsSpecial(seed, n, 2);
        crsMatrix cm_2 = newCrsSpecial(seed, n, 2);

        regMatrix rm_1 = regFromCrs(&cm_1);
        regMatrix rm_2 = regFromCrs(&cm_2);
    
        float time;
        regMatrix rm_o = regMulReg(rm_1, rm_2, &time);
        crsMatrix cm_o = crsMulCrs(cm_1, cm_2, &time);

        int res = checkEq(cm_o, rm_o);
        if (res == 0) {
            printf("For n=%d is ok\n", n);
        }
    }

    // Task 21.
    printf("\nRegular matrix-matrix multiplication serial:\n");
    testRegMulReg();
    for (int cores = 2; cores <= 8; cores *= 2) {
        printf("\nRegular matrix-matrix multiplication with OMP with %d core(s):\n", cores);
        omp_set_num_threads(cores);
        testRegMulRegOmp();
    }

    //Task 23.
    printf("\nCrs matrix-matrix multiplication serial:\n");
    testCrsMulCrs();
    for (int cores = 2; cores <= 8; cores *= 2) {
        printf("\nCrs matrix-matrix multiplication with OMP with %d core(s):\n", cores);
        omp_set_num_threads(cores);
        testCrsMulCrsOmp();
    }


    printf("\nTask 25. Checking matrix multiplication on GPU\n"); 
    for (int n = 10; n <= 1000; n *= 10) {
        float time;
        crsMatrix cm = newCrsSpecial(seed, n, 2);
        regMatrix rm = regFromCrs(&cm);

        vector_decl(float, x);
        for (int i = 0; i < n; i++){
            vector_push(&x, n / 2 - i);
        }
        vector_decl(float, gpu_result);
        gpu_result = regMulVecGpu(rm, x, &time);

        vector_decl(float, reg_result);
        regMultVector(&rm, &x, &reg_result, &time);
        
        printf("For n = %d: ", n);
        float diff = compareVectors(&gpu_result, &reg_result, n);
        if (fabs(diff) < ZERO_IN_REG){
            printf("correct");
        } else{
            printf("incorrect: %f", diff);
        }
        printf("\n");
    }

    
    printf("\nReg matrix-vector multiplication on GPU:\n");
    testRegMulVecGpu();

    printf("\nTask 28. Checking CRS matrix multiplication on GPU\n"); 
    for (int n = 10; n <= 1000; n *= 10) {
        float time;
        crsMatrix cm = newCrsSpecial(seed, n, 2);
        regMatrix rm = regFromCrs(&cm);

        vector_decl(float, x);
        for (int i = 0; i < n; i++){
            vector_push(&x, n / 2 - i);
        }
        vector_of(float) gpu_result = crsMulVecGpu(cm, x, &time);

        vector_decl(float, reg_result);
        regMultVector(&rm, &x, &reg_result, &time);
        
        printf("For n = %d: ", n);
        float diff = compareVectors(&gpu_result, &reg_result, n);
        if (fabs(diff) < ZERO_IN_REG){
            printf("correct");
        } else{
            printf("incorrect: %f", diff);
        }
        printf("\n");
    }

    printf("\nCrs matrix-vector multiplication on GPU:\n");    
    testCrsMulVecGpu();


    printf("\nTask 28. Checking CRS matrix multiplication on GPU\n"); 
    for (int n = 10; n <= 1000; n *= 10) {
        float time;
        crsMatrix cm_1 = newCrsSpecial(seed, n, 2);
        crsMatrix cm_2 = newCrsSpecial(seed, n, 2);

        regMatrix a = regFromCrs(&cm_1);
        regMatrix b = regFromCrs(&cm_2);

        regMatrix c_gpu = regMulRegGpu(a, b, &time);
        regMatrix c_real = regMulReg(a, b, &time);

        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++) {
                float diff = c_gpu.values[i][j] - c_real.values[i][j];
                DIE(diff > ZERO_IN_REG);
            }
        }
        printf("For n=%d is ok\n", n);
    }
    

    return 0;
}
