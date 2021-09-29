#include "im2col.h"
#include "trace.h"
#include <omp.h>
#include <immintrin.h>

void im2col(Tensor* im, const int kernel_size, const int pad, const int stride, Tensor* col) {
    MAKE_SURE(im->c * kernel_size * kernel_size == col->w, "NOT MATCHED.");

    int conv2d_h = (im->h + 2 * pad - kernel_size) / stride + 1;
    int conv2d_w = (im->w + 2 * pad - kernel_size) / stride + 1;

    int K = im->c;
    int J = im->c * kernel_size;
    int I = im->c * kernel_size * kernel_size;
    int N = im->c * kernel_size * kernel_size * conv2d_w;
    int K_ = im->c;
    int J_ = im->c * im->w;
    int I_ = im->c * stride;
    int N_ = im->c * im->w * stride;

    // #pragma omp parallel for
    for (int n = 0; n < conv2d_h; ++n) {
        for (int i = 0; i < conv2d_w; ++i) {
            for (int j = 0; j != kernel_size; ++j) {
                for (int k = 0; k != kernel_size; ++k) {
                    for (int v = 0; v != im->c; ++v) {
                        // printf("%d\n", v + k * K_ + j * J_ + i * I_ + n * N_);
                        col->data[v + k * K + j * J + i * I + n * N] = im->data[v + k * K_ + j * J_ + i * I_ + n * N_];
                    }
                }
            }
        }
    }
}