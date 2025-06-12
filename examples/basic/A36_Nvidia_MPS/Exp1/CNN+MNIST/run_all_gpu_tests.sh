#!/bin/bash

# Run CPU-only version
EXPERIMENT_NAME=cpu CUDA_VISIBLE_DEVICES="" python3 train.py

# Run full GPU version
EXPERIMENT_NAME=gpu_100 python3 train.py

# Run MPS limited versions
for pct in 5; do
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct
    ./processor_count
    EXPERIMENT_NAME=gpu_$pct python3 train.py
done

# Combine CSVs and plot
python3 plot_all.py
