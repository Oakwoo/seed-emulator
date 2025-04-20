#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
using namespace std;

int main(){
    cudaSetDevice(0);
    struct cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("cudaDevAttrMultiProcessorCount: %d\n\n", devProp.multiProcessorCount);
    return 0;
}
