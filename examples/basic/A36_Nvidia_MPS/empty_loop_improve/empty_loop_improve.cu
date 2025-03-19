#include <stdio.h>
#include <stdlib.h>

#define MAX_DELAY 30

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(__err), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    } \
  } while (0)


#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

unsigned long long dtime_usec(unsigned long long start){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

__global__ void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++);
    //y[i] = x[i] + y[i];
}

int main(int argc, char *argv[]){

  int N = 1<<25;
  float *x, *y;
  // Allocate Unified Memory - accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f; y[i] = 2.0f;
  }
  unsigned long long difft = dtime_usec(0);
  // Run kernel on 1M elements on the GPU
  add<<<1, 1>>>(N, x, y);
  cudaDeviceSynchronize();
  cudaCheckErrors("kernel fail");
  difft = dtime_usec(difft);
  printf("kernel duration: %fs\n", difft/(float)USECPSEC);
  return 0;
}
