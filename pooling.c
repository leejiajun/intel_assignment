#include "pooling.h"

#include <immintrin.h>
#include <omp.h>

#include "im2col.h"

// DType
void max_pool2d_k2s2(Tensor* ifmap, Tensor* ofmap) {
    int stride = 2;
    int I = ofmap->c * ifmap->w * stride;
    int J = ofmap->c * stride;
    int R = ofmap->c;             // right-shift one element
    int D = ofmap->c * ifmap->w;  // down-shift one element

    for (int i = 0; i < ofmap->h; i++) {
        for (int j = 0; j < ofmap->w; j++) {
            for (int k = 0; k < ofmap->c; k++) {
                int id0 = k + j * J + i * I;
                int id1 = R + k + j * J + i * I;
                int id2 = D + k + j * J + i * I;
                int id3 = R + D + k + j * J + i * I;
                ofmap->data[k + j * ofmap->c + i * ofmap->c * ofmap->w] = fmaxf(fmaxf(ifmap->data[id0], ifmap->data[id1]), fmaxf(ifmap->data[id2], ifmap->data[id3]));
            }
        }
    }
}

int max_index_4(DType a, DType b, DType c, DType d) {
    if ((a >= b) && (a >= c) && (a >= d)) {
        return 0;
    } else if ((b >= a) && (b >= c) && (b >= d)) {
        return 1;
    } else if ((c >= a) && (c >= b) && (c >= d)) {
        return 2;
    } else if ((d >= a) && (d >= b) && (d >= c)) {
        return 3;
    } else {
        return -1;
    }
}

void max_pool2d_k2s2_backward(Tensor* grad_in, Tensor* cache_ifmap, Tensor* grad_out) {
    int stride = 2;
    int I = grad_in->c * grad_out->w * stride;
    int J = grad_in->c * stride;
    int R = grad_in->c;                // right-shift one element
    int D = grad_in->c * grad_out->w;  // down-shift one element

    set_tensor_zeros(grad_out);

    for (int i = 0; i < grad_in->h; i++) {
        for (int j = 0; j < grad_in->w; j++) {
            for (int k = 0; k < grad_in->c; k++) {
                int id0 = k + j * J + i * I;
                int id1 = R + k + j * J + i * I;
                int id2 = D + k + j * J + i * I;
                int id3 = R + D + k + j * J + i * I;
                int max_idx = max_index_4(cache_ifmap->data[id0], cache_ifmap->data[id1], cache_ifmap->data[id2], cache_ifmap->data[id3]);
                if (max_idx == 0) {
                    grad_out->data[id0] = grad_in->data[k + j * grad_in->c + i * grad_in->c * grad_in->w];
                } else if (max_idx == 1) {
                    grad_out->data[id1] = grad_in->data[k + j * grad_in->c + i * grad_in->c * grad_in->w];
                } else if (max_idx == 2) {
                    grad_out->data[id2] = grad_in->data[k + j * grad_in->c + i * grad_in->c * grad_in->w];
                } else if (max_idx == 3) {
                    grad_out->data[id3] = grad_in->data[k + j * grad_in->c + i * grad_in->c * grad_in->w];
                }
            }
        }
    }
}

void max_pool2d_k2s2_omp(Tensor* ifmap, Tensor* ofmap) {
    int stride = 2;
    int I = ofmap->c * ifmap->w * stride;
    int J = ofmap->c * stride;
    int R = ofmap->c;             // right-shift one element
    int D = ofmap->c * ifmap->w;  // down-shift one element

#pragma omp parallel for
    for (int i = 0; i < ofmap->h; i++) {
        for (int j = 0; j < ofmap->w; j++) {
            for (int k = 0; k < ofmap->c; k++) {
                int id0 = k + j * J + i * I;
                int id1 = R + k + j * J + i * I;
                int id2 = D + k + j * J + i * I;
                int id3 = R + D + k + j * J + i * I;
                ofmap->data[k + j * ofmap->c + i * ofmap->c * ofmap->w] = fmaxf(fmaxf(ifmap->data[id0], ifmap->data[id1]), fmaxf(ifmap->data[id2], ifmap->data[id3]));
            }
        }
    }
}

float _mm256_max_vec4_ps(const float* lt, const float* rt, const float* lb, const float* rb, const int N, float* maximun) {
    for (int i = 0; i < N / 8; ++i) {
        __m256 v0 = _mm256_loadu_ps(lt + 8 * i);
        __m256 v1 = _mm256_loadu_ps(rt + 8 * i);
        __m256 v2 = _mm256_loadu_ps(lb + 8 * i);
        __m256 v3 = _mm256_loadu_ps(rb + 8 * i);
        __m256 z =_mm256_max_ps(_mm256_max_ps(v0, v1), _mm256_max_ps(v2, v3));
        _mm256_storeu_ps(maximun + i * 8, z);
    }
    if (N % 8 != 0) {
        for (int i = (N / 8) * 8; i < N; ++i) {
            maximun[i] = fmaxf(fmaxf(lt[i], rt[i]), fmaxf(lb[i], rb[i]));
        }
    }
}

void max_pool2d_k2s2_simd_omp(Tensor* ifmap, Tensor* ofmap) {
    int stride = 2;
    int I = ofmap->c * ifmap->w * stride;
    int J = ofmap->c * stride;
    int R = ofmap->c;             // right-shift one element
    int D = ofmap->c * ifmap->w;  // down-shift one element

#pragma omp parallel for
    for (int i = 0; i < ofmap->h; i++) {
        for (int j = 0; j < ofmap->w; j++) {
            int id0 = j * J + i * I;
            int id1 = R + j * J + i * I;
            int id2 = D + j * J + i * I;
            int id3 = R + D + j * J + i * I;
            int out_offset = j * ofmap->c + i * ofmap->c * ofmap->w;
            _mm256_max_vec4_ps(ifmap->data + id0, ifmap->data + id1, ifmap->data + id2, ifmap->data + id3, ofmap->c, (float*) ofmap->data + out_offset);
        }
    }
}