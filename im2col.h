#ifndef __IM2COL_H__
#define __IM2COL_H__

#include "tensor.h"

void im2col(Tensor* im, const int kernel_size, const int pad, const int stride, Tensor* col);

#endif /* __IM2COL_H__ */