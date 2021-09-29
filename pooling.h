#ifndef __POOLING_H__
#define __POOLING_H__

#include "tensor.h"

void max_pool2d_k2s2(Tensor* ifmap, Tensor* ofmap);

void max_pool2d_k2s2_backward(Tensor* grad_in, Tensor* cache_ifmap, Tensor* grad_out);

void max_pool2d_k2s2_omp(Tensor* ifmap, Tensor* ofmap);

void max_pool2d_k2s2_simd_omp(Tensor* ifmap, Tensor* ofmap);

#endif /* __POOLING_H__ */