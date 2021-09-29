#ifndef EXCEPTIONS
#define EXCEPTIONS

#include <err.h>
#include <execinfo.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void print_stacktrace(int calledFromSigInt);

#define STATUS_OK 1
#define MAKE_SURE(ret, msg)                                                                        \
    if (ret != STATUS_OK) {                                                                        \
        print_stacktrace(0);                                                                       \
        fprintf(stderr, "\"%s\" %s (in %s at %s:%i).\n", #ret, msg, __func__, __FILE__, __LINE__); \
        exit(1);                                                                                   \
    }

#define MAX_BACKTRACE_LINES 64

#endif
