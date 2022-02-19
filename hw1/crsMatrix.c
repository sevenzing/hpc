#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define VECTOR_IMPLEMENTATION
#include "vector.h"

#define MAX_VAL 100


typedef struct {
    // Matrix size (N x N)
    int n;
    // Number of non-zero elements
    int nz;
    // Array of values (size NZ)
    vector_of(float) value;
    // Array of column numbers (size NZ)
    vector_of(int) col;
    // Array of row indexes (size N + 1)
    vector_of(int) row_index;
} crsMatrix;



crsMatrix newCrs(int N, int NZ) {
    vector_decl(float, values);
    for (int i = 0; i < NZ; i++){
      vector_push(&values, 0);
    };

    vector_decl(int, col);
    for (int i = 0; i < NZ; i++){
      vector_push(&col, 0);
    };
    
    vector_decl(int, row_index);
    for (int i = 0; i < N + 1; i++){
      vector_push(&row_index, 0);
    };

    return (crsMatrix){N, NZ, values, col, row_index};
}

float next() {
  return ((float)rand() / (float)RAND_MAX);
}

crsMatrix newCrsRandom(int seed, int N, int cntInRow) {
  int i, j, k, f, tmp, notNull, c;

  srand(seed);

  notNull = cntInRow * N;
  crsMatrix mtx = newCrs(N, notNull);

  for(i = 0; i < N; i++)
  {
    for(j = 0; j < cntInRow; j++)
    {
      do
      {
        mtx.col[i * cntInRow + j] = rand() % N;
        f = 0;
        for (k = 0; k < j; k++)
          if (mtx.col[i * cntInRow + j] == mtx.col[i * cntInRow + k])
            f = 1;
      } while (f == 1);
    }
    for (j = 0; j < cntInRow - 1; j++)
      for (k = 0; k < cntInRow - 1; k++)
        if (mtx.col[i * cntInRow + k] > mtx.col[i * cntInRow + k + 1])
        {
          tmp = mtx.col[i * cntInRow + k];
          mtx.col[i * cntInRow + k] = mtx.col[i * cntInRow + k + 1];
          mtx.col[i * cntInRow + k + 1] = tmp;
        }
  }
  
  for (i = 0; i < notNull; i++){
    mtx.value[i] = next() * MAX_VAL;
  }
  c = 0;
  for (i = 0; i <= N; i++) {
    mtx.row_index[i] = c;
    c += cntInRow;
  }

  return mtx;
}


crsMatrix newCrsSpecial(int seed, int N, int cntInRow){
  srand(seed);
  float end = pow((float)cntInRow, 1.0 / 3.0);
  float step = end / N;

  vector_decl(vector_of(int), columns);
  int NZ = 0;

  for (int i = 0; i < N; i++){
    vector_push(&columns, NULL);
    int rowNZ = pow(((i + 1) * step), 3) + 1;
    NZ += rowNZ;
    int num1 = (rowNZ - 1) / 2;
    int num2 = rowNZ - 1 - num1;

    if (rowNZ != 0){
      if (i < num1){
        num2 += num1 - i;
        num1 = i;
        for(int j = 0; j < i; j++){
          vector_push(&columns[i], j);
        }
        vector_push(&columns[i], i);
        
        for(int j = 0; j < num2; j++){
          vector_push(&columns[i], i + 1 + j);
        }
      }
      else
      {
        if (N - i - 1 < num2){
          num1 += num2 - (N - 1 - i);
          num2 = N - i - 1;
        }
        for (int j = 0; j < num1; j++){ 
          vector_push(&columns[i], i - num1 + j);
        }
        vector_push(&(columns[i]), i);

        
        for (int j = 0; j < num2; j++){
          vector_push(&columns[i], i + j + 1);
        }
      }
    }
  }
  crsMatrix mtx = newCrs(N, NZ);

  int count = 0;
  int sum = 0;
  for (int i = 0; i < N; i++)
  {
    mtx.row_index[i] = sum;
    sum += vector_len(&columns[i]);
    for (unsigned int j = 0; j < vector_len(&columns[i]); j++)
    {
      mtx.col[count] = columns[i][j];
      mtx.value[count] = next();
      count++;
    }
  }
  mtx.row_index[N] = sum;

  vector_iter(&columns, _, vector_of(int) v, {
      vector_free(&v);
  });
  vector_free(&columns);
  return mtx;
}


void printCrs(crsMatrix *mtx) {
  vector_iter(&mtx->row_index, row, int columns_index_start, {
    if ((int)row == mtx->n) {
      break;
    }
    int columnds_index_end = mtx->row_index[row + 1];
    int j = 0;
    for (int column_index = columns_index_start; column_index < columnds_index_end; column_index++){
      int column = mtx->col[column_index];
      while (j < column){
        printfloat(0);
        j++;
      }
      printfloat(mtx->value[column_index]);
      j++;
    }
    while (j < mtx->n) {
      printfloat(0);
      j++;
    }
    printf("\n");
  });
}


void freeCrs(crsMatrix *mtx){
  vector_free(&mtx->col);
  vector_free(&mtx->row_index);
  vector_free(&mtx->value);
}

void copyCrs(crsMatrix im, crsMatrix *om){
    
}