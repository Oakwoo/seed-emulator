#!/usr/bin/env python3
# encoding: utf-8

import torch
devProp = torch.cuda.get_device_properties(torch.device('cuda'))
print("cudaDevAttrMultiProcessorCount:", devProp.multi_processor_count)
