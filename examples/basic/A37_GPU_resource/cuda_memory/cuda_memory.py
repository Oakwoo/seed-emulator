#!/usr/bin/env python3
# encoding: utf-8

import torch
free_t, total_t = torch.cuda.mem_get_info()
free_m =free_t/1048576.0;
total_m=total_t/1048576.0;
used_m=total_m-free_m;
print("memory free", free_t, "....", free_m, "MB");
print("memory total", total_t, "....", total_m, "MB");
print("memory used", used_m, "MB");

