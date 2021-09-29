#include <immintrin.h>
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>

#include "conv2d.h"
#include "im2col.h"
#include "pooling.h"
#include "rdtsc.h"
#include "relu.h"

void print_3d_map(Tensor* map);

void print_4d_kernel(Tensor* kernel);

void ReLUTestCase(int type);

void MaxPoolTestCase();

void Conv2dTestCase(int type);

void IM2COLTestCase();

void MATMULTestCase();

void func_performance(int runs, void (*func)());

int leak_test(char* para);

void conv_performance(char* name, int runs, void (*conv_func)(), int var);

void relu_performance(char* name, int runs, void (*relu_func)(), int var);

void pooling_performance(char* name, int runs, void (*pooling_func)(), int var);

int main(int argc, char** argv) {
    IM2COLTestCase();
    MATMULTestCase();
    Conv2dTestCase(0);
    Conv2dTestCase(1);
    Conv2dTestCase(2);
    Conv2dTestCase(3);
    MaxPoolTestCase(0);
    MaxPoolTestCase(1);
    MaxPoolTestCase(2);
    ReLUTestCase(0);
    ReLUTestCase(1);
    ReLUTestCase(2);
    ReLUTestCase(3);

    int runs = 10;
    conv_performance("conv2d", runs, conv2d, 256);
    conv_performance("conv2d_omp", runs, conv2d_omp, 256);
    conv_performance("conv2d_omp_im2col_locality", runs, conv2d_omp_im2col_locality, 256);
    conv_performance("conv2d_simd_fma_omp_im2col_locality", runs, conv2d_simd_fma_omp_im2col_locality, 256);

    relu_performance("relu", runs, relu, 256);
    relu_performance("relu_omp", runs, relu_omp, 256);
    relu_performance("relu_simd_omp", runs, relu_simd_omp, 256);
    relu_performance("relu_simd_unrolling", runs, relu_simd_unrolling, 256);

    pooling_performance("max_pool2d_k2s2", runs, max_pool2d_k2s2, 256);
    pooling_performance("max_pool2d_k2s2_omp", runs, max_pool2d_k2s2_omp, 256);
    pooling_performance("max_pool2d_k2s2_simd_omp", runs, max_pool2d_k2s2_simd_omp, 256);

    // for (int i = 1; i < 32; ++i) {
    //     conv_performance("conv2d", runs, conv2d, 16 * i);
    //     conv_performance("conv2d_omp", runs, conv2d_omp, 16 * i);
    //     conv_performance("conv2d_omp_im2col_locality", runs, conv2d_omp_im2col_locality, 16 * i);
    //     conv_performance("conv2d_simd_fma_omp_im2col_locality", runs, conv2d_simd_fma_omp_im2col_locality, 16 * i);
    // }

    // for (int i = 1; i < 32; ++i) {
    //     relu_performance("relu", runs, relu, 16 * i);
    //     relu_performance("relu_omp", runs, relu_omp, 16 * i);
    //     relu_performance("relu_simd_omp", runs, relu_simd_omp, 16 * i);
    //     relu_performance("relu_simd_unrolling", runs, relu_simd_unrolling, 16 * i);
    // }

    // for (int i = 1; i < 32; ++i) {
    //     pooling_performance("max_pool2d_k2s2", runs, max_pool2d_k2s2, 16 * i);
    //     pooling_performance("max_pool2d_k2s2_omp", runs, max_pool2d_k2s2_omp, 16 * i);
    //     pooling_performance("max_pool2d_k2s2_simd_omp", runs, max_pool2d_k2s2_simd_omp, 16 * i);
    // }

    leak_test("leaking_testing");
}

void conv_performance(char* name, int runs, void (*conv_func)(), int var) {
    int input_channel = 16;
    int output_channel = 32;
    int input_h = var;
    int input_w = 64;
    int kernel_size = 3;
    int stride = 1;
    int padding = 0;  // not working
    int output_h = floor((input_h - kernel_size) / stride) + 1;
    int output_w = floor((input_w - kernel_size) / stride) + 1;

    int M = kernel_size * kernel_size * input_channel;
    int N = output_h * output_w;
    int K = output_channel;
    double gflop = (2.0 * M * N * K) * 1E-9;

    Tensor kernel, ifmap, ofmap;
    init_tensor_rand(&kernel, input_channel, output_channel, kernel_size * kernel_size);
    init_tensor_rand(&ifmap, input_channel, input_h, input_w);
    init_tensor_rand(&ofmap, output_channel, output_h, output_w);

    double cycle = 0;
    tsc_counter t0, t1;
    double time = 0.0;
    double duration;

    RDTSC(t0);
    for (unsigned int i = 0; i != runs; ++i) {
        (*conv_func)(&kernel, kernel_size, ifmap.c, ofmap.c, stride, &ifmap, &ofmap);
    }
    RDTSC(t1);
    cycle = COUNTER_DIFF(t1, t0, CYCLES);
    time = COUNTER_DIFF(t1, t0, SEC);
    duration = (double)(time / ((double)runs));

    printf("\nName: %s\n", name);
    printf("Average cycle : %.1f\n", ((double)(cycle / ((double)runs))));
    printf("Average second: %e\n", duration);
    printf("GFlop         : %.4f\n", gflop);
    printf("GFlop/s       : %.4f\n", gflop / duration);

    free_tensor(&ifmap);
    free_tensor(&ofmap);
    free_tensor(&kernel);
}

void relu_performance(char* name, int runs, void (*relu_func)(), int var) {
    int input_channel = 16;
    int output_channel = input_channel;
    int input_h = var;
    int input_w = 640;
    int output_h = input_h;
    int output_w = input_w;
    double gflop = (input_channel * input_h * input_w) * 1E-9;

    Tensor ifmap;
    init_tensor_rand(&ifmap, input_channel, input_h, input_w);

    double cycle = 0;
    tsc_counter t0, t1;
    double time = 0.0;
    double duration;

    RDTSC(t0);
    for (unsigned int i = 0; i != runs; ++i) {
        (*relu_func)(&ifmap);
    }
    RDTSC(t1);
    cycle = COUNTER_DIFF(t1, t0, CYCLES);
    time = COUNTER_DIFF(t1, t0, SEC);
    duration = (double)(time / ((double)runs));

    printf("\nName: %s\n", name);
    printf("Average cycle : %.1f\n", ((double)(cycle / ((double)runs))));
    printf("Average second: %e\n", duration);
    printf("GFlop         : %.4f\n", gflop);
    printf("GFlop/s       : %.4f\n", gflop / duration);

    free_tensor(&ifmap);
}

void pooling_performance(char* name, int runs, void (*pooling_func)(), int var) {
    int input_channel = 64;
    int output_channel = 64;
    int input_h = var;
    int input_w = 512;
    int kernel_size = 2;
    int stride = 2;
    int output_h = floor((input_h - kernel_size) / stride) + 1;
    int output_w = floor((input_w - kernel_size) / stride) + 1;

    // int M = kernel_size * kernel_size * input_channel;
    // int N = output_h * output_w;
    // int K = output_channel;
    double gflop = (3.0 * input_channel * input_h * input_w / 4.0) * 1E-9;

    Tensor ifmap, ofmap;
    init_tensor_rand(&ifmap, input_channel, input_h, input_w);
    init_tensor_rand(&ofmap, output_channel, output_h, output_w);

    double cycle = 0;
    tsc_counter t0, t1;
    double time = 0.0;
    double duration;

    RDTSC(t0);
    for (unsigned int i = 0; i != runs; ++i) {
        (*pooling_func)(&ifmap, &ofmap);
    }
    RDTSC(t1);
    cycle = COUNTER_DIFF(t1, t0, CYCLES);
    time = COUNTER_DIFF(t1, t0, SEC);
    duration = (double)(time / ((double)runs));

    printf("\nName: %s\n", name);
    printf("Average cycle : %.1f\n", ((double)(cycle / ((double)runs))));
    printf("Average second: %e\n", duration);
    printf("GFlop         : %.4f\n", gflop);
    printf("GFlop/s       : %.4f\n", gflop / duration);

    free_tensor(&ifmap);
    free_tensor(&ofmap);
}

void ReLUTestCase(int type) {
    // CHECK const __attribute__((aligned(32)))
    DType ifmap_data[60] = {9.0, -4.0, 0.0, 1.0, 3.0, 2.0, -3.0, 4.0, 2.0, 0.0, 0.0, 1.0, -2.0, 7.0, 8.0,
                            1.0, -3.0, 4.0, 5.0, 9.0, 2.0, 3.0, -2.0, 7.0, 0.0, 0.0, 5.0, 8.0, 5.0, 5.0,
                            3.0, 7.0, 4.0, -3.0, 3.0, 5.0, 8.0, -4.0, 0.0, 2.0, 3.0, 5.0, 0.0, -3.0, 0.0,
                            9.0, 0.0, 8.0, 8.0, -3.0, 4.0, -5.0, 9.0, 3.0, 9.0, 8.0, 2.0, -8.0, -5.0, 4.0};
    DType ofmap_data[60] = {0};
    DType ans[60] = {9.0, 0.0, 0.0, 1.0, 3.0, 2.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0, 0.0, 7.0, 8.0,
                     1.0, 0.0, 4.0, 5.0, 9.0, 2.0, 3.0, 0.0, 7.0, 0.0, 0.0, 5.0, 8.0, 5.0, 5.0,
                     3.0, 7.0, 4.0, 0.0, 3.0, 5.0, 8.0, 0.0, 0.0, 2.0, 3.0, 5.0, 0.0, 0.0, 0.0,
                     9.0, 0.0, 8.0, 8.0, 0.0, 4.0, 0.0, 9.0, 3.0, 9.0, 8.0, 2.0, 0.0, 0.0, 4.0};

    Tensor ifmap;
    init_tensor(&ifmap, 3, 4, 5, ifmap_data);

    Tensor ofmap;
    init_tensor(&ofmap, 3, 4, 5, ofmap_data);

    if (type == 0) {
        relu(&ifmap);
    } else if (type == 1) {
        relu_omp(&ifmap);
    } else if (type == 2) {
        relu_simd_omp(&ifmap);
    } else if (type == 3) {
        relu_simd_unrolling(&ifmap);
    }

    int CHECK = 1;
    for (int i = 0; i != ifmap.c * ifmap.h * ifmap.w; i++) {
        // if (type == 1) printf("%f:%f\n", ifmap.data[i], ans[i]);
        if (ifmap.data[i] != ans[i]) {
            CHECK = 0;
        }
    }

    if (CHECK == 1) {
        printf("[CHECK PASS]: ");
    } else if (CHECK == 0) {
        printf("[CHECK FAILED]: ");
    }

    if (type == 0) {
        printf("ReLU.\n");
    } else if (type == 1) {
        printf("ReLU OMP.\n");
    } else if (type == 2) {
        printf("ReLU SIMD+OMP.\n");
    } else if (type == 3) {
        printf("ReLU SIMD+UNROLLING.\n");
    }
}

void MaxPoolTestCase(int type) {
    DType ifmap_data[60] = {9.0, 4.0, 0.0, 1.0, 3.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 1.0, 2.0, 7.0, 8.0,
                            1.0, 3.0, 4.0, 5.0, 9.0, 2.0, 3.0, 2.0, 7.0, 0.0, 0.0, 5.0, 8.0, 5.0, 5.0,
                            3.0, 7.0, 4.0, 3.0, 3.0, 5.0, 8.0, 4.0, 0.0, 2.0, 3.0, 5.0, 0.0, 3.0, 0.0,
                            9.0, 0.0, 8.0, 8.0, 3.0, 4.0, 5.0, 9.0, 3.0, 9.0, 8.0, 2.0, 8.0, 5.0, 4.0};
    DType ofmap_data[12] = {0};

    DType ans[12] = {9.0, 9.0, 4.0, 3.0, 4.0, 7.0,
                     9.0, 7.0, 8.0, 9.0, 9.0, 5.0};

    // kernel
    const int kernel_size = 2;
    const int stride = 2;
    const int padding = 0;

    Tensor ifmap, ofmap;
    init_tensor(&ifmap, 3, 4, 5, ifmap_data);
    init_tensor(&ofmap, 3, 2, 2, ofmap_data);

    if (type == 0) {
        max_pool2d_k2s2(&ifmap, &ofmap);
    } else if (type == 1) {
        max_pool2d_k2s2_omp(&ifmap, &ofmap);
    } else if (type == 2) {
        max_pool2d_k2s2_simd_omp(&ifmap, &ofmap);
    }

    int CHECK = 1;
    for (int i = 0; i != ofmap.c * ofmap.h * ofmap.w; i++) {
        // if (type == 3) printf("%f:%f\n", ofmap.data[i], ans[i]);
        if (ofmap.data[i] != ans[i]) {
            CHECK = 0;
        }
    }

    if (CHECK == 1) {
        printf("[CHECK PASS]: ");
    } else if (CHECK == 0) {
        printf("[CHECK FAILED]: ");
    }

    if (type == 0) {
        printf("Max Pooling.\n");
    } else if (type == 1) {
        printf("Max Pooling OMP.\n");
    } else if (type == 2) {
        printf("Max Pooling OMP+SIMD.\n");
    }
}

void Conv2dTestCase(int type) {
    // CHECK
    DType ifmap_data[60] = {9.0, 4.0, 0.0, 1.0, 3.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 1.0,
                            2.0, 7.0, 8.0, 1.0, 3.0, 4.0, 5.0, 9.0, 2.0, 3.0, 2.0, 7.0,
                            0.0, 0.0, 5.0, 8.0, 5.0, 5.0, 3.0, 7.0, 4.0, 3.0, 3.0, 5.0,
                            8.0, 4.0, 0.0, 2.0, 3.0, 5.0, 0.0, 3.0, 0.0, 9.0, 0.0, 8.0,
                            8.0, 3.0, 4.0, 5.0, 9.0, 3.0, 9.0, 8.0, 2.0, 8.0, 5.0, 4.0};

    DType ofmap_data[18] = {0};
    DType kernel_data[81] = {1.0, 3.0, 5.0, 6.0, 4.0, 2.0, 7.0, 0.0, 3.0,
                             7.0, 3.0, 8.0, 1.0, 2.0, 8.0, 3.0, 4.0, 8.0,
                             2.0, 3.0, 6.0, 5.0, 8.0, 8.0, 8.0, 2.0, 6.0,

                             0.0, 4.0, 8.0, 4.0, 6.0, 1.0, 5.0, 9.0, 4.0,
                             1.0, 2.0, 7.0, 1.0, 5.0, 0.0, 3.0, 9.0, 7.0,
                             3.0, 4.0, 4.0, 7.0, 6.0, 7.0, 3.0, 2.0, 5.0,

                             0.0, 6.0, 1.0, 3.0, 8.0, 4.0, 0.0, 2.0, 8.0,
                             1.0, 6.0, 9.0, 0.0, 3.0, 1.0, 2.0, 0.0, 1.0,
                             3.0, 6.0, 7.0, 8.0, 6.0, 8.0, 9.0, 7.0, 0.0};

    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 0;

    // ifmap
    Tensor ifmap;
    init_tensor(&ifmap, 3, 4, 5, ifmap_data);

    // ofmap
    Tensor ofmap;
    init_tensor(&ofmap, 3, 2, 3, ofmap_data);

    // kernel
    Tensor kernel;  // (channel_input, channel_output, kernel_size * kernel_size)
    init_tensor(&kernel, 3, 3, 9, kernel_data);

    ifmap.data = ifmap_data;
    ofmap.data = ofmap_data;
    kernel.data = kernel_data;

    DType ans[18] = {432.0, 419.0, 441.0, 411.0, 313.0, 361.0,
                     380.0, 411.0, 355.0, 543.0, 513.0, 593.0,
                     511.0, 458.0, 561.0, 531.0, 476.0, 468.0};

    if (type == 0) {
        conv2d(&kernel, kernel_size, ifmap.c, ofmap.c, stride, &ifmap, &ofmap);
    } else if (type == 1) {
        conv2d_omp(&kernel, kernel_size, ifmap.c, ofmap.c, stride, &ifmap, &ofmap);
    } else if (type == 2) {
        conv2d_omp_im2col_locality(&kernel, kernel_size, ifmap.c, ofmap.c, stride, &ifmap, &ofmap);
    } else if (type == 3) {
        conv2d_simd_fma_omp_im2col_locality(&kernel, kernel_size, ifmap.c, ofmap.c, stride, &ifmap, &ofmap);
    }

    int CHECK = 1;
    for (int i = 0; i != ofmap.c * ofmap.h * ofmap.w; i++) {
        // if (type == 3) printf("%f:%f\n", ofmap.data[i], ans[i]);
        if (ofmap.data[i] != ans[i]) {
            CHECK = 0;
        }
    }

    if (CHECK == 1) {
        printf("[CHECK PASS]: ");
    } else if (CHECK == 0) {
        printf("[CHECK FAILED]: ");
    }

    if (type == 0) {
        printf("Conv2d.\n");
    } else if (type == 1) {
        printf("Conv2d OMP.\n");
    } else if (type == 2) {
        printf("Conv2d OMP+IM2COL+LOCALITY.\n");
    } else if (type == 3) {
        printf("Conv2d SIMD+FMA+OMP+IM2COL+LOCALITY.\n");
    }
}

void IM2COLTestCase() {
    // CHECK
    DType im_data[60] = {9.0, 4.0, 0.0, 1.0, 3.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 1.0, 2.0, 7.0, 8.0,
                         1.0, 3.0, 4.0, 5.0, 9.0, 2.0, 3.0, 2.0, 7.0, 0.0, 0.0, 5.0, 8.0, 5.0, 5.0,
                         3.0, 7.0, 4.0, 3.0, 3.0, 5.0, 8.0, 4.0, 0.0, 2.0, 3.0, 5.0, 0.0, 3.0, 0.0,
                         9.0, 0.0, 8.0, 8.0, 3.0, 4.0, 5.0, 9.0, 3.0, 9.0, 8.0, 2.0, 8.0, 5.0, 4.0};
    DType col_data[162] = {0};  // (kernel_size * kernel_size * im_c) * (ofmap_c * ofmap_h * ofmap_w)

    DType ans[162] = {9.0, 4.0, 0.0, 1.0, 3.0, 2.0, 3.0, 4.0, 2.0,
                      1.0, 3.0, 4.0, 5.0, 9.0, 2.0, 3.0, 2.0, 7.0,
                      3.0, 7.0, 4.0, 3.0, 3.0, 5.0, 8.0, 4.0, 0.0,

                      1.0, 3.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 1.0,
                      5.0, 9.0, 2.0, 3.0, 2.0, 7.0, 0.0, 0.0, 5.0,
                      3.0, 3.0, 5.0, 8.0, 4.0, 0.0, 2.0, 3.0, 5.0,

                      3.0, 4.0, 2.0, 0.0, 0.0, 1.0, 2.0, 7.0, 8.0,
                      3.0, 2.0, 7.0, 0.0, 0.0, 5.0, 8.0, 5.0, 5.0,
                      8.0, 4.0, 0.0, 2.0, 3.0, 5.0, 0.0, 3.0, 0.0,

                      1.0, 3.0, 4.0, 5.0, 9.0, 2.0, 3.0, 2.0, 7.0,
                      3.0, 7.0, 4.0, 3.0, 3.0, 5.0, 8.0, 4.0, 0.0,
                      9.0, 0.0, 8.0, 8.0, 3.0, 4.0, 5.0, 9.0, 3.0,

                      5.0, 9.0, 2.0, 3.0, 2.0, 7.0, 0.0, 0.0, 5.0,
                      3.0, 3.0, 5.0, 8.0, 4.0, 0.0, 2.0, 3.0, 5.0,
                      8.0, 3.0, 4.0, 5.0, 9.0, 3.0, 9.0, 8.0, 2.0,

                      3.0, 2.0, 7.0, 0.0, 0.0, 5.0, 8.0, 5.0, 5.0,
                      8.0, 4.0, 0.0, 2.0, 3.0, 5.0, 0.0, 3.0, 0.0,
                      5.0, 9.0, 3.0, 9.0, 8.0, 2.0, 8.0, 5.0, 4.0};

    // im
    Tensor im;
    init_tensor(&im, 3, 4, 5, im_data);

    // kernel
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 0;

    // col
    Tensor col;
    init_tensor(&col, 1, 6, 27, col_data);

    im2col(&im, kernel_size, padding, stride, &col);
    // conv2d_omp(kernel, kernel_size, ifmap_c, ofmap_c, stride, ifmap, ifmap_h, ifmap_w, ofmap, ofmap_h, ofmap_w, 2);

    int CHECK = 1;
    for (int i = 0; i != col.c * col.h * col.w; i++) {
        // printf("%f:%f\n", col.data[i], ans[i]);
        if (col.data[i] != ans[i]) {
            CHECK = 0;
            // break;
        }
    }

    if (CHECK == 1) {
        printf("[CHECK PASS]: IM2COL.\n");
    } else if (CHECK == 0) {
        printf("[CHECK FAILED]: IM2COL.\n");
    }
}

void MATMULTestCase() {
    // Tensor A;
    // init_tensor(&A, 1, 2, 3);
    // Tensor B;
    // init_tensor(&B, 1, 3, 3);
    // Tensor C;
    // init_tensor(&C, 1, 2, 3);

    // // 1x6x27 chw
    // DType A_data[6] = {9.0, 4.0, 0.0,
    //                    1.0, 3.0, 4.0};
    // // 27x3
    // DType B_data[9] = {1.0, 0.0, 0.0,
    //                    3.0, 4.0, 6.0,
    //                    5.0, 8.0, 1.0};
    // DType C_data[6] = {0};

    // A.data = A_data;
    // B.data = B_data;
    // C.data = C_data;

    // DType ans[6] = {21.0, 16.0, 24.0,
    //                 30.0, 44.0, 22.0};

    // 1x6x27 chw
    DType A_data[162] =
        {9.0, 4.0, 0.0, 1.0, 3.0, 2.0, 3.0, 4.0, 2.0,
         1.0, 3.0, 4.0, 5.0, 9.0, 2.0, 3.0, 2.0, 7.0,
         3.0, 7.0, 4.0, 3.0, 3.0, 5.0, 8.0, 4.0, 0.0,

         1.0, 3.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 1.0,
         5.0, 9.0, 2.0, 3.0, 2.0, 7.0, 0.0, 0.0, 5.0,
         3.0, 3.0, 5.0, 8.0, 4.0, 0.0, 2.0, 3.0, 5.0,

         3.0, 4.0, 2.0, 0.0, 0.0, 1.0, 2.0, 7.0, 8.0,
         3.0, 2.0, 7.0, 0.0, 0.0, 5.0, 8.0, 5.0, 5.0,
         8.0, 4.0, 0.0, 2.0, 3.0, 5.0, 0.0, 3.0, 0.0,

         1.0, 3.0, 4.0, 5.0, 9.0, 2.0, 3.0, 2.0, 7.0,
         3.0, 7.0, 4.0, 3.0, 3.0, 5.0, 8.0, 4.0, 0.0,
         9.0, 0.0, 8.0, 8.0, 3.0, 4.0, 5.0, 9.0, 3.0,

         5.0, 9.0, 2.0, 3.0, 2.0, 7.0, 0.0, 0.0, 5.0,
         3.0, 3.0, 5.0, 8.0, 4.0, 0.0, 2.0, 3.0, 5.0,
         8.0, 3.0, 4.0, 5.0, 9.0, 3.0, 9.0, 8.0, 2.0,

         3.0, 2.0, 7.0, 0.0, 0.0, 5.0, 8.0, 5.0, 5.0,
         8.0, 4.0, 0.0, 2.0, 3.0, 5.0, 0.0, 3.0, 0.0,
         5.0, 9.0, 3.0, 9.0, 8.0, 2.0, 8.0, 5.0, 4.0};
    // 27x3
    DType B_data[81] = {1.0, 0.0, 0.0,
                        3.0, 4.0, 6.0,
                        5.0, 8.0, 1.0,
                        6.0, 4.0, 3.0,
                        4.0, 6.0, 8.0,
                        2.0, 1.0, 4.0,
                        7.0, 5.0, 0.0,
                        0.0, 9.0, 2.0,
                        3.0, 4.0, 8.0,
                        7.0, 1.0, 1.0,
                        3.0, 2.0, 6.0,
                        8.0, 7.0, 9.0,
                        1.0, 1.0, 0.0,
                        2.0, 5.0, 3.0,
                        8.0, 0.0, 1.0,
                        3.0, 3.0, 2.0,
                        4.0, 9.0, 0.0,
                        8.0, 7.0, 1.0,
                        2.0, 3.0, 3.0,
                        3.0, 4.0, 6.0,
                        6.0, 4.0, 7.0,
                        5.0, 7.0, 8.0,
                        8.0, 6.0, 6.0,
                        8.0, 7.0, 8.0,
                        8.0, 3.0, 9.0,
                        2.0, 2.0, 7.0,
                        6.0, 5.0, 0.0};
    DType C_data[18] = {0};

    Tensor A;
    init_tensor(&A, 1, 6, 27, A_data);
    Tensor B;
    init_tensor(&B, 1, 27, 3, B_data);
    Tensor C;
    init_tensor(&C, 1, 6, 3, C_data);

    DType ans[18] = {432.0, 419.0, 441.0, 411.0, 313.0, 361.0,
                     380.0, 411.0, 355.0, 543.0, 513.0, 593.0,
                     511.0, 458.0, 561.0, 531.0, 476.0, 468.0};

    mat_mul(&A, &B, &C);
    // conv2d_omp(kernel, kernel_size, ifmap_c, ofmap_c, stride, ifmap, ifmap_h, ifmap_w, ofmap, ofmap_h, ofmap_w, 2);

    int CHECK = 1;
    for (int i = 0; i != C.c * C.h * C.w; i++) {
        // printf("%f:%f\n", C.data[i], ans[i]);
        if (C.data[i] != ans[i]) {
            CHECK = 0;
            // break;
        }
    }

    if (CHECK == 1) {
        printf("[CHECK PASS]: MAT_MUL.\n");
    } else if (CHECK == 0) {
        printf("[CHECK FAILED]: MAT_MUL.\n");
    }
}

void func_performance(int runs, void (*func)()) {
    long long sum1 = 0;
    tsc_counter t0, t1;
    for (unsigned int i = 0; i != runs; ++i) {
        RDTSC(t0);
        (*func)();
        RDTSC(t1);
        sum1 += (COUNTER_DIFF(t1, t0, CYCLES));
    }
    printf("Average time: %lf cycles\n", ((double)(sum1 / ((double)runs))));
}

int leak_test(char* para) {
    if (NULL == para) {
        //local_log("LeakTest Func: empty parameter\n");
        return -1;
    }
    char* log_msg = malloc(128 * sizeof(char));
    if (NULL == log_msg) {
        //local_log("memeory allocation failed\n");
        return -2;
    }
    sprintf(log_msg, "LeakTest routine exit: '%s'.\n", para);
    //local_log(Logmsg);
    return 0;
}

void print_3d_map(Tensor* map) {
    for (int c = 0; c < map->c; c++) {
        for (int h = 0; h < map->h; h++) {
            for (int w = 0; w < map->w; w++) {
                // printf("%d ", C*w + C*W*h + c);
                printf("%f ", map->data[c + w * map->c + h * map->c * map->w]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_4d_kernel(Tensor* kernel) {
    int COUT = kernel->h;
    int CIN = kernel->c;
    int SIZE = (int)sqrt(kernel->w);
    for (int i = 0; i < COUT; i++) {
        for (int j = 0; j < CIN; j++) {
            for (int k = 0; k < SIZE; k++) {
                for (int v = 0; v < SIZE; v++) {
                    // printf("%d ", j + v * SIZE + k * CIN * SIZE + i * CIN * SIZE * SIZE);
                    printf("%f ", kernel->data[j + v * SIZE + k * CIN * SIZE + i * CIN * SIZE * SIZE]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}