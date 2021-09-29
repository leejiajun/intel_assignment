# intel_assignment

## running

``` shell
sh run.sh
```
or
``` shell
gcc -std=gnu99 -mavx2 -mfma -mfma4 -fopenmp -lm trace.c tensor.c im2col.c conv2d.c pooling.c relu.c main.c -o main.o && ./main.o
```
