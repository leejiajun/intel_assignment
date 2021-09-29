#include "relu.h"

#include <omp.h>
#include <immintrin.h>

void relu(Tensor* ifmap) {
    for (int i = 0; i != ifmap->c * ifmap->h * ifmap->w; i++) {
        ifmap->data[i] = fmaxf((DType)0.0, ifmap->data[i]);
    }
}

void relu_backward(Tensor* activation, Tensor* back) {
    for (int i = 0; i != activation->c * activation->h * activation->w; i++) {
        back->data[i] *= activation->data[i] <= 0.0f ? 0.0f : 1.0f;
    }
}

void relu_omp(Tensor* ifmap) {
#pragma omp parallel num_threads(1)
    for (int i = 0; i < ifmap->c * ifmap->h * ifmap->w; i++) {
        ifmap->data[i] = fmaxf((DType)0.0, ifmap->data[i]);
    }
}

void relu_simd_omp(Tensor* ifmap) {
    int N = ifmap->c * ifmap->h * ifmap->w;

    const __m256 zero = _mm256_set1_ps(0.0f);
    __m256 YMM;

// #pragma omp parallel for
    for (int i = 0; i < N / 8; ++i) {
        YMM = _mm256_loadu_ps(ifmap->data + i * 8);
        YMM = _mm256_max_ps(zero, YMM);
        _mm256_storeu_ps(ifmap->data + i * 8, YMM);
    }

    if (N % 8 != 0) {
        for (int i = (N / 8) * 8; i < N; ++i) {
            ifmap->data[i] = fmaxf((DType)0.0, ifmap->data[i]);
        }
    }
}

void relu_simd_unrolling(Tensor* ifmap) {
    int N = ifmap->c * ifmap->h * ifmap->w;

    const __m256 zero = _mm256_set1_ps(0.0f);

    __m256 YMM0, YMM1;

    for (int i = 0; i <= ((N)-16); i += 16) {
        YMM0 = _mm256_load_ps(ifmap->data + i);
        YMM1 = _mm256_load_ps(ifmap->data + i + 8);
        YMM0 = _mm256_max_ps(zero, YMM0);
        YMM1 = _mm256_max_ps(zero, YMM1);
        _mm256_store_ps(ifmap->data + i, YMM0);
        _mm256_store_ps(ifmap->data + i + 8, YMM1);
    }

    if (N % 16 != 0) {
        for (int i = (N / 16) * 16; i < N; ++i) {
            ifmap->data[i] = fmaxf((DType)0.0, ifmap->data[i]);
        }
    }
}
