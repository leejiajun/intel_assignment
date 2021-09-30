# intel_assignment
This source is for the interviewing of INTEL.

## running
``` shell
sh run.sh
```
or
``` shell
gcc -std=gnu99 -mavx2 -mfma -mfma4 -fopenmp -lm \
    trace.c tensor.c im2col.c conv2d.c pooling.c relu.c main.c \
    -o main.o \
    && ./main.o
```

## tensor storage
In my design, every matrix is corresponding to a one-dimensional array, as shown in Figure 1. Given a 3-D matrix with a 3x4x5 shape on the left side, my program stores each element into memory along channel direction, to form a sequential one-dimensional space on the right side of Figure 1. To a 4-D matrix, the storage way almost goes like a 3-D matrix, as shown in Figure 2. I already put the index on the element square, and hope it can help to understand.

<div align=center>
  <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/leejiajun/intel_assignment/blob/main/imgs/1.png" width="50%">
  <br>
  <div align=center style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
      Figure 1. 3-D Tensor Storage.
  </div>
  <br>
</div>

<div align=center>
  <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/leejiajun/intel_assignment/blob/main/imgs/2.png" width="70%">
  <br>
  <div align=center style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
      Figure 2. 4-D Tensor Storage.
  </div>
  <br>
</div>

Specially, I use only one struct to represent the 3-D matrix and 4-D matrix, as shown in the following code block. It is easy to understand that this struct represents a 3-D matrix. However, in my design, a 4-D matrix is also stored in this struct, which means the 4-D matrix needs to be represented internally. As shown in Figure 3, the OUTSIDE format is what you can see, representing by my Tensor struct, where the shape is \[Input Channel, Output Channel, Kernel Size^2\], a 3-D dimensional shape. Meanwhile, the internal shape of this 4-D matrix is \[Input Channel, Output Channel, Kernel Size, Kernel Size\], which needs your imagination. Their storage sequences in memory are exactly the same.

<div align=center>
  <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/leejiajun/intel_assignment/blob/main/imgs/3.png" width="70%">
  <br>
  <div align=center style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
      Figure 3. Internal Representing of 4-D Matrix by My Struct.
  </div>
  <br>
</div>


## output example

```shell
Name: conv2d
Average cycle : 859606612.6
Average second: 3.737420e-01
GFlop         : 0.0720
GFlop/s       : 0.1926

Name: conv2d_omp
Average cycle : 123192591.4
Average second: 5.356200e-02
GFlop         : 0.0720
GFlop/s       : 1.3442

Name: conv2d_omp_im2col_locality
Average cycle : 67256949.2
Average second: 2.924215e-02
GFlop         : 0.0720
GFlop/s       : 2.4620

Name: conv2d_simd_fma_omp_im2col_locality
Average cycle : 55251988.8
Average second: 2.402260e-02
GFlop         : 0.0720
GFlop/s       : 2.9970

Name: relu
Average cycle : 132116989.4
Average second: 5.744217e-02
GFlop         : 0.0131
GFlop/s       : 0.2282

Name: relu_omp
Average cycle : 161830332.8
Average second: 7.036101e-02
GFlop         : 0.0131
GFlop/s       : 0.1863

Name: relu_simd_omp
Average cycle : 22298009.4
Average second: 9.694787e-03
GFlop         : 0.0131
GFlop/s       : 1.3520

Name: relu_simd_unrolling
Average cycle : 21837244.6
Average second: 9.494454e-03
GFlop         : 0.0131
GFlop/s       : 1.3805

Name: max_pool2d_k2s2
Average cycle : 78454523.6
Average second: 3.411066e-02
GFlop         : 0.0063
GFlop/s       : 0.1844

Name: max_pool2d_k2s2_omp
Average cycle : 52021273.8
Average second: 2.261795e-02
GFlop         : 0.0063
GFlop/s       : 0.2782

Name: max_pool2d_k2s2_simd_omp
Average cycle : 51135153.8
Average second: 2.223268e-02
GFlop         : 0.0063
GFlop/s       : 0.2830
```

## reference
- [1] https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/
- [2] https://github.com/BVLC/caffe
- [3] https://github.com/pytorch/pytorch
- [4] https://software.intel.com/content/www/us/en/develop/articles/a-simple-example-to-measure-the-performance-of-an-intel-mkl-function.html
- [5] https://arxiv.org/pdf/1808.05567.pdf
- [6] https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction
- [7] https://www.intel.com/content/dam/support/us/en/documents/processors/APP-for-Intel-Xeon-Processors.pdf

## to-do
Mat Mul with Tilling
