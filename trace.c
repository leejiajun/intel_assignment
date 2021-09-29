#include "trace.h"

void print_stacktrace(int calledFromSigInt) {
    void *buffer[MAX_BACKTRACE_LINES];
    char **strings;

    int nptrs = backtrace(buffer, MAX_BACKTRACE_LINES);
    strings = backtrace_symbols(buffer, nptrs);
    if (strings == NULL) {
        perror("backtrace_symbols");
        exit(EXIT_FAILURE);
    }

    unsigned int i = 1;
    if (calledFromSigInt != 0)
        ++i;
    for (; i < (unsigned int)(nptrs - 2); ++i) {
        fprintf(stderr, "[%i] %s\n", nptrs - 2 - i - 1, strings[i]);
    }

    free(strings);
}
