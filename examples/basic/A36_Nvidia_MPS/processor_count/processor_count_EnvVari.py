#!/usr/bin/env python3
# encoding: utf-8
import os
# Set the environment variable;
# this environment variable is only valid within the current process.
os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']='5'

import torch
devProp = torch.cuda.get_device_properties(torch.device('cuda'))
print("cudaDevAttrMultiProcessorCount:", devProp.multi_processor_count)
