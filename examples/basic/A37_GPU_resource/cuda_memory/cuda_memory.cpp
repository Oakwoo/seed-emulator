#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
using namespace std;

int main(){
    float free_m,total_m,used_m;
    size_t free_t,total_t;
    cudaMemGetInfo(&free_t,&total_t);
    free_m =(float)free_t/1048576.0;
    total_m=(float)total_t/1048576.0;
    used_m=total_m-free_m;
    printf ( "memory free %ld .... %f MB\nmemory total %ld....%f MB\nmemory used %f MB\n",free_t,free_m,total_t,total_m,used_m);
    return 0;
}
