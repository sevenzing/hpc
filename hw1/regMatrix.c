#include <stdio.h>
#include <stdlib.h>

#define VECTOR_IMPLEMENTATION
#include "vector.h"

typedef struct {
    int n;
    float** values;
} regMatrix;



regMatrix newReg(int N){
    float** values = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++) {
        values[i] = (float*) malloc(N * sizeof(float));
        for (int j = 0; j < N; j++){
          values[i][j] = 0.0;
        }
    }
    return (regMatrix){N, values};
}

void freeReg(regMatrix *mtx){
  for (int i = 0; i < mtx->n; i++) {
        free(mtx->values[i]);
  }
  free(mtx->values);
}

float* get_flatten(regMatrix m){
  float* v = (float*)malloc(m.n * m.n * sizeof(float));
  for (int i = 0; i < m.n; i++){
    for (int j = 0; j < m.n; j++) {
      v[i * m.n + j] = m.values[i][j];
    }
  }
  return v;
}


void printReg(regMatrix *mtx) {
  for (int i = 0; i < mtx->n; i++){
    for (int j = 0; j < mtx->n; j++){
      printfloat(mtx->values[i][j]);
    }
    printf("\n");
  }
}
