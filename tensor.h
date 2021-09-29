#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <math.h>
#include <time.h>
#include <stdlib.h>

typedef float DType;

typedef struct {
    int c, h, w;
    /* guarantee that mat will not be aliased (__restrict),
    no need for two matrices to point at sama data */
    DType* __restrict data;
} Tensor;

void init_tensor(Tensor* t, int c, int h, int w, const DType* init_data);

void init_tensor_empty(Tensor* t, int c, int h, int w);

void init_tensor_zeros(Tensor* t, int c, int h, int w);

void init_tensor_rand(Tensor* t, int c, int h, int w);

void set_tensor_zeros(Tensor* t);

void free_tensor(Tensor* t);

#endif /* __TENSOR_H__ */