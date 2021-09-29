#include "conv2d.h"
#include "trace.h"
#include <omp.h>
#include <immintrin.h>
#include "im2col.h"

/* in_channel*ifmap_h*ifmap_w */
/* out_channel*ofmap_h*ofmap_w */
void conv2d(Tensor* kernel, int kernel_size, int in_channel, int out_channel, int stride, Tensor* ifmap, Tensor* ofmap) {
    // CHECK: kernel_h and kernel_w must be odd numbers.
    // CHECK: ofmap_h must equal (ifmap_h - kernel_size + 1) / stride + 1
    // CHECK: ofmap_w must equal (ifmap_w - kernel_size + 1) / stride + 1
    for (int k = 0; k < out_channel; k++) {
        for (int i = 0; i < ofmap->h; i++) {
            for (int j = 0; j < ofmap->w; j++) {                // sum
                for (int ki = 0; ki < kernel_size; ki++) {      // h. partial sum
                    for (int kj = 0; kj < kernel_size; kj++) {  // w. partial partial sum
                        for (int kk = 0; kk < in_channel; kk++) {
                            // printf("%d ", j + (i * ofmap_w) + (k * ofmap_h * ofmap_w));
                            // printf("%d:%d ", kk + kj * in_channel + ki * in_channel * ifmap_w + j * in_channel * stride + i * in_channel * ifmap_w * stride,
                            //kk + kj * in_channel + ki * in_channel * kernel_size + k * in_channel * kernel_size * kernel_size);
                            ofmap->data[k + (i * ofmap->w * out_channel) + (j * out_channel)] += ifmap->data[kk + kj * in_channel + ki * in_channel * ifmap->w + j * in_channel * stride + i * in_channel * ifmap->w * stride] * kernel->data[kk + kj * in_channel + ki * in_channel * kernel_size + k * in_channel * kernel_size * kernel_size];
                            // ofmap->data[j + (i * ofmap->w) + (k * ofmap->h * ofmap->w)] += ifmap->data[kk + kj * in_channel + ki * in_channel * ifmap->w + j * in_channel * stride + i * in_channel * ifmap->w * stride] * kernel[kk + kj * in_channel + ki * in_channel * kernel_size + k * in_channel * kernel_size * kernel_size];
                        }
                    }
                }
            }
        }
    }
}

void conv2d_omp(Tensor* kernel, int kernel_size, int in_channel, int out_channel, int stride, Tensor* ifmap, Tensor* ofmap) {
// CHECK: kernel_h and kernel_w must be odd numbers.
// CHECK: ofmap_h must equal (ifmap_h - kernel_size + 1) / stride + 1
// CHECK: ofmap_w must equal (ifmap_w - kernel_size + 1) / stride + 1
#pragma omp parallel for
    for (int k = 0; k < out_channel; k++) {
        for (int i = 0; i < ofmap->h; i++) {
            for (int j = 0; j < ofmap->w; j++) {                // sum
                for (int ki = 0; ki < kernel_size; ki++) {      // h. partial sum
                    for (int kj = 0; kj < kernel_size; kj++) {  // w. partial partial sum
                        for (int kk = 0; kk < in_channel; kk++) {
                            ofmap->data[k + (i * ofmap->w * out_channel) + (j * out_channel)] += ifmap->data[kk + kj * in_channel + ki * in_channel * ifmap->w + j * in_channel * stride + i * in_channel * ifmap->w * stride] * kernel->data[kk + kj * in_channel + ki * in_channel * kernel_size + k * in_channel * kernel_size * kernel_size];
                        }
                    }
                }
            }
        }
    }
}

void col2im(Tensor* col) {
}

void kernel_im2col(Tensor* kernel) {
    kernel->w *= kernel->c;
    kernel->c = 1;
}

/*
MN * NK = MK
*/
void mat_mul(Tensor* A, Tensor* B, Tensor* C) {
    MAKE_SURE(A->w == B->h, "NOT MATCHED");
    MAKE_SURE(A->h == C->h, "NOT MATCHED");
    MAKE_SURE(B->w == C->w, "NOT MATCHED");

    int M = A->h;
    int N = A->w;
    int K = B->w;

// #pragma omp simd reduction(+:res) aligned(A, x, y:32)
#pragma omp simd
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N; ++n) {
                // printf("%f\n", A->data[n + m * N]);
                // printf("%d %d %d\n", n*K + k, n + m * N, m);
                C->data[k + m * K] += A->data[n + m * N] * B->data[n * K + k];
            }
        }
    }
}

/* B mat is transposed. It should be MN * KN = MK */
void mat_mul_A_BT(Tensor* A, Tensor* B, Tensor* C) {
    MAKE_SURE(A->w == B->w, "NOT MATCHED");
    MAKE_SURE(A->h == C->h, "NOT MATCHED");
    MAKE_SURE(B->w == C->w, "NOT MATCHED");

    int M = A->h;
    int N = A->w;
    int K = B->w;

    // #pragma omp simd reduction(+:res) aligned(A, x, y:32)
    // #pragma omp simd
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N; ++n) {
                C->data[k + m * K] += A->data[n + m * N] * B->data[n + k * N];
            }
        }
    }
}

void conv2d_omp_im2col_locality(Tensor* kernel, int kernel_size, int in_channel, int out_channel, int stride, Tensor* ifmap, Tensor* ofmap) {
    Tensor col;
    init_tensor_empty(&col, 1, ofmap->h * ofmap->w, kernel_size * kernel_size * ifmap->c);
    im2col(ifmap, kernel_size, 0, stride, &col);  // 1x6x27
    kernel_im2col(kernel);                        // 1x3x27

    int M = col.h;
    int N = col.w;
    int K = kernel->h;

#pragma omp parallel for
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N; ++n) {
                ofmap->data[k + m * K] += col.data[n + m * N] * kernel->data[n + k * N];
            }
        }
    }
}

float _mm256_reduce_add_ps(__m256 v) {
    // ( x3, x2, x1, x0 ) + ( x7, x6, x5, x4 ) = ( x3+x7, x2+x6, x1+x5, x0+x4 )
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    // ( x3+x7, x2+x6, x1+x5, x0+x4) + ( x2+x6, x2+x6, x0+x4, x0+x4 ) = ( x3+x7+x2+x6, x2+x6+x2+x6, x1+x5+x0+x4, x0+x4+x0+x4 )
    const __m128 x64 = _mm_add_ps(x128, _mm_movehdup_ps(x128));
    // ( x3+x7+x2+x6, x2+x6+x2+x6, x1+x5+x0+x4, x0+x4+x0+x4 ) + ( x1+x5+x0+x4, x0+x4+x0+x4, x1+x5, x0+x4 )
    const __m128 x32 = _mm_add_ss(x64, _mm_movehl_ps(x128, x64));
    // ( x2+x6+x3+x7+x1+x5+x0+x4, _, _, _ )
    return _mm_cvtss_f32(x32);
}

float _mm256_dot_product_ps(const float* a, const float* b, const int N) {
    // for (int i = 0; i < N; ++i) {
    //     printf("%f %f %f\n", a[i], b[i], N);
    // }
    // int len_a = sizeof(a) / sizeof(a[0]);
    // int len_b = sizeof(b) / sizeof(b[0]);
    // printf("%d %d %d\n", len_a, len_b, N);
    // MAKE_SURE(len_a == len_b, "NOT MATCHED");
    // MAKE_SURE(len_a == N, "NOT MATCHED");

    __m256 sum_vec = _mm256_setzero_ps();  //_mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    /* Add up partial dot-products in blocks of 256 bits */
    for (int i = 0; i < N / 8; ++i) {
        __m256 x = _mm256_loadu_ps(a + 8 * i);
        __m256 y = _mm256_loadu_ps(b + 8 * i);
        __m256 z = _mm256_mul_ps(x, y);
        sum_vec = _mm256_add_ps(sum_vec, z);
    }

    float left = 0.0;
    if (N % 8 != 0) {
        for (int i = (N / 8) * 8; i < N; ++i) {
            left += a[i] * b[i];
        }
    }

    return _mm256_reduce_add_ps(sum_vec) + left;
}

// float _mm512_reduce_add_ps(__m512 v) {
//     // ( x7, x6, x5, x4, x3, x2, x1, x0 ) + ( x15, x14, x13, x12, x11, x10, x9, x8 ) =
//     const __m256 up = _mm512_extractf256_ps(v, 1);
//     const __m256 down = _mm512_castps512_ps256(v);
//     const __m256 x256 = _mm256_add_ps(_mm512_extractf256_ps(v, 1), _mm512_castps512_ps256(v));
//     return _mm256_reduce_add_ps(x256);
// }

// float _mm512_reduce_add_ps(__m512 a) {
//     __m512 temp1_AVX_512 = _mm512_shuffle_f32x4(a, a, 0x4e);  //swap high and low halves of a
//     __m512 temp2_AVX_512 = _mm512_add_ps(a, temp1_AVX_512);   // sum corresponding floats in the high, low halves

//     temp1_AVX_512 = _mm512_shuffle_f32x4(temp2_AVX_512, temp2_AVX_512, 0xb1);  //swap high and low quarters of each half of temp2
//     temp2_AVX_512 = _mm512_add_ps(temp2_AVX_512, temp1_AVX_512);               // sum corresponding floats in the high, low quarters

//     temp1_AVX_512 = _mm512_shuffle_ps(temp2_AVX_512, temp2_AVX_512, 0x4e);  //swap high and low eigths of each quarter of a
//     temp2_AVX_512 = _mm512_add_ps(temp2_AVX_512, temp1_AVX_512);            // sum corresponding floats in the high, low eighths

//     temp1_AVX_512 = _mm512_shuffle_ps(temp2_AVX_512, temp2_AVX_512, 0xb1);  //swap high and low sixteenths of each eighth
//     temp2_AVX_512 = _mm512_add_ps(temp2_AVX_512, temp1_AVX_512);            // each element of temp2_AVX_512 now contains the sum of all the floats in a

//     __m256 temp3_AVX = _mm512_extractf32x8_ps(temp2_AVX_512, 0);  //Grab the low half of temp2_AVX_512
//                                                                   // because AVX-512 doesn't provide an operation to extract one float from a 512-bit vector
//                                                                   // printf("output sum vector is: ");
//                                                                   // print_512(temp2_AVX_512);
//     //   int *retint_ptr = (int *) ret_sum;  // This is a horrible hack because there isn't an intrinsic to extract a float from
//     //   // an __m256.  Do this to avoid casting an int back to a float and screwing it up
//     //   *retint_ptr = _mm256_extract_epi32((__m256i) temp3_AVX, 0);
//     return (float)_mm256_extract_epi32((__m256i)temp3_AVX, 0);
// }

// float _mm512_dot_product_ps(const float* a, const float* b, const int N) {
//     __m512 sum_vec = _mm512_setzero_ps();

//     /* Add up partial dot-products in blocks of 256 bits */
//     for (int i = 0; i < N / 16; ++i) {
//         __m512 x = _mm512_loadu_ps(a + 16 * i);
//         __m512 y = _mm512_loadu_ps(b + 16 * i);
//         __m512 z = _mm512_mul_ps(x, y);
//         sum_vec = _mm512_add_ps(sum_vec, z);
//     }

//     float left = 0.0;
//     if (N % 8 != 0) {
//         for (int i = (N / 8) * 8; i < N; ++i) {
//             left += a[i] * b[i];
//         }
//     }

//     return _mm512_reduce_add_ps(sum_vec) + left;
// }

void conv2d_simd_fma_omp_im2col_locality(Tensor* kernel, int kernel_size, int in_channel, int out_channel, int stride, Tensor* ifmap, Tensor* ofmap) {
    Tensor col;
    init_tensor_empty(&col, 1, ofmap->h * ofmap->w, kernel_size * kernel_size * ifmap->c);
    im2col(ifmap, kernel_size, 0, stride, &col);  // 1x6x27
    kernel_im2col(kernel);                        // 1x3x27

    int M = col.h;
    int N = col.w;
    int K = kernel->h;

#pragma omp parallel for
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            ofmap->data[k + m * K] = (DType)_mm256_dot_product_ps(col.data + m * N, kernel->data + k * N, N);
        }
    }
}


