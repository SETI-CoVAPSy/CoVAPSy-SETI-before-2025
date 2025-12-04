#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define N 10000000

int32_t set1[N], set2[N], result[N];
struct timespec start, end, start_filing, end_filing;
long duration;

int main() {

    // Generate the sets
    
    printf("Filling C array\n");
    clock_gettime(CLOCK_MONOTONIC, &start_filing);
    for (int i = 0; i < N; i++) {
        set2[i] = abs(rand()) % 200 + 1; // Ensure no division by zero
        set1[i] = (69420 + i) * set2[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &end_filing);
    duration = (end_filing.tv_sec - start_filing.tv_sec) * 1e9 + (end.tv_nsec - start_filing.tv_nsec);
    printf("Fommed %d array in %ld\n", N, duration);

    // Measure the division time
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < N; i++) {
        result[i] = set1[i] / set2[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate duration in nanoseconds
    duration = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);

    // Print the results
    printf("Division results:\n");
    for (int i = 0; i < 3; i++) {
        printf("result[%d] = %d\n", i, result[i]);
    }   
    printf("Time taken for division: %ld nanoseconds\n", duration);
    printf("Division per second in C: %f\n", ((float) N)/(((float) duration)/1E9));

    return 0;
}