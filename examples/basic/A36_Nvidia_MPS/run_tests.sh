#!/bin/bash

for percent in 100 75 50 25
do
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$percent
    echo "Running with active_thread = $percent%"
    ./kernel_test
    echo ""
done
