#ifndef __RELU_H__
#define __RELU_H__

#include "tensor.h"

void relu(Tensor* ifmap);

void relu_omp(Tensor* ifmap);

void relu_simd_omp(Tensor* ifmap);

void relu_simd_unrolling(Tensor* ifmap);

void relu_backward(Tensor* activation, Tensor* back);

#endif /* __RELU_H__ */