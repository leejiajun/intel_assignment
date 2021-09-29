#include "tensor.h"

// const int ALIGN = 64;

void init_tensor(Tensor* t, int c, int h, int w, const DType* init_data) {
    t->c = c;
    t->h = h;
    t->w = w;
    // t->data = init_data;
    posix_memalign((void**)&t->data, 64, c * h * w * sizeof(DType));
    for (int i = 0; i < c * h * w; ++i) {
        t->data[i] = init_data[i];
    }
}

void init_tensor_empty(Tensor* t, int c, int h, int w) {
    t->c = c;
    t->h = h;
    t->w = w;
    posix_memalign((void**)&t->data, 64, c * h * w * sizeof(DType));
}

void init_tensor_zeros(Tensor* t, int c, int h, int w) {
    t->c = c;
    t->h = h;
    t->w = w;
    posix_memalign((void**)&t->data, 64, c * h * w * sizeof(DType));
    for (int i = 0; i < c * h * w; ++i) {
        t->data[i] = 0.0;
    }
}

void init_tensor_rand(Tensor* t, int c, int h, int w) {
    t->c = c;
    t->h = h;
    t->w = w;
    // posix_memalign((void**) t->data, 64, c * h * w * sizeof(DType));
    // t->data = malloc(c * h * w * sizeof(DType));
    posix_memalign((void**)&t->data, 64, c * h * w * sizeof(DType));
    srand((int)time(NULL));
    for (int i = 0; i != c * h * w; i++) {
        t->data[i] = (DType)(rand() % 100 / 16.0);
    }
}

void set_tensor_zeros(Tensor* t) {
    for (int i = 0; i < t->c * t->h * t->w; ++i) {
        t->data[i] = 0.0;
    }
}

void free_tensor(Tensor* t) {
    if (t->data != NULL) {
        free(t->data);
    }
    t->data = NULL;
}