#ifndef __CONV2D_H__
#define __CONV2D_H__

#include "tensor.h"

void conv2d(Tensor* kernel, int kernel_size, int in_channel, int out_channel, int stride, Tensor* ifmap, Tensor* ofmap);

void conv2d_omp(Tensor* kernel, int kernel_size, int in_channel, int out_channel, int stride, Tensor* ifmap, Tensor* ofmap);

void conv2d_omp_im2col_locality(Tensor* kernel, int kernel_size, int in_channel, int out_channel, int stride, Tensor* ifmap, Tensor* ofmap);

void conv2d_simd_fma_omp_im2col_locality(Tensor* kernel, int kernel_size, int in_channel, int out_channel, int stride, Tensor* ifmap, Tensor* ofmap);

void mat_mul(Tensor* A, Tensor* B, Tensor* C);

#endif /* __CONV2D_H__ */