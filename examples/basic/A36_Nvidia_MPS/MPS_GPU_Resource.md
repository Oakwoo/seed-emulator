# Multi-Process Service control GPU hardware resources
When your GPU?s compute capacity exceeds the needs of a single application, running multiple processes that share the same GPU can be beneficial. In the [previous tutorial](./MPS_basic_tutorial.md), we introduced how NVIDIA MPS (Multi-Process Service) controls concurrent access to GPUs by multiple independent processes. When multiple independent processes share the GPU, it?s often useful to set customized resource allocation limits to prevent any single process from consuming excessive GPU resource.
In **NVIDIA Volta and newer architecture GPU**, Nvidia MPS provides finer-grain control over GPU hardware resources, including CUDA SMs(Streaming Multiprocessors), GPU memory.

In this tutorial, we will systematically introduce how to set hardware limits for GPU MPS clients.


## Get familiar with device's resource

Before setting limitations, we first need to understand our device's resources. By default, the program utilizes the entire GPU resources.

### Check the Number of Streaming Multiprocessors (SMs)
we can use either Python or C++ code prepared in `./processor_count` folder to print out the number of Streaming Multiprocessors (SMs) on an Nvidia GPU.
* **python**:
```shell-script
$ python3 processor_count.py
cudaDevAttrMultiProcessorCount: 40
```

* **C++**:
```shell-script
$ nvcc processor_count.cpp -o processor_count
$ ./processor_count
cudaDevAttrMultiProcessorCount: 40
```

The result shows that the Nvidia Tesla T4 GPU has a total of 40 SMs.

### Check the GPU Memory Usage
we can use either Python or C++ code prepared in `./cuda_memory` folder to print out the memory usage on an Nvidia GPU.
* **python**:
```shell-script
$ python3 cuda_memory.py
memory free 15520563200 .... 14801.5625 MB
memory total 15655829504 .... 14930.5625 MB
memory used 129.0 MB
```

* **C++**:
```shell-script
$ nvcc cuda_memory.cpp -o cuda_memory
$ ./cuda_memory
memory free 15520563200 .... 14801.562500 MB
memory total 15655829504....14930.562500 MB
memory used 129.000000 MB
```

The result shows that the Nvidia Tesla T4 GPU has a total of 14.58GB memory.

#### TODO *Memory Footprint
To provide a per-thread stack, CUDA reserves 1kB of GPU memory per thread
This is (2048 threads per SM x 1kB per thread) = 2 MB per SM used, or 164 MB per client for V100 (221 MB for A100)
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE reduces max SM usage, and so reduces memory footprint
Each MPS process also uploads a new copy of the executable code, which adds to the memory footprint


## Restrict CUDA Threads ceils? Streaming Multiprocessors
streaming processor(sp): ?????????streaming processor ?????????????sp?????GPU?????????????sp????????SP?????????????????thread??????SP????thread?



### Start MPS Daemon
To use MPS to restrict GPU compute resources, we first need to start the Nvidia control daemon using following commands:
```shell-script
# create the folder for MPS pipes and log files if necessary
$ mkdir /tmp/mps_0
$ mkdir /tmp/mps_log_0

# set the environment variables.
$ export CUDA_VISIBLE_DEVICES=0
$ export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_0
$ export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_0

# set the GPU's compute mode and start the MPS control daemon
$ sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
Set compute mode to EXCLUSIVE_PROCESS for GPU 00000000:00:1E.0.
All done.
$ nvidia-cuda-mps-control -d
```


### Set the Default Active Thread Limits
By default, the `default active thread percentage` is set to 100%, meaning MPS clients can use the entire GPU streaming processor resource. The `set_default_active_thread_percentage <percentage>` command overrides the default active thread percentage. All the MPS servers spawned by the MPS control daemon will observe this limit.
```shell-script
$ echo "set_default_active_thread_percentage 10" | nvidia-cuda-mps-control
10.0
```
 In this case, we set the client contexts to use up to `10%` of the available threads.

### Verify if the limits are valid
We can use the same program as before to check if the limits are valid.

**Here's how it works:**  Once the MPS client of `processor_count.py` connects to the MPS control daemon, an MPS server is created that enforces the active thread limit, restricting `processor_count.py` to use only 10% of the available threads.


```shell-script
$ python3 processor_count.py
cudaDevAttrMultiProcessorCount: 4
```
In this case, the number of available SMs reduces from 40 to 4, which is 10% of the total GPU resources, confirming that the limitations are working. :tada:

### Update the limit for the existing server

#### :warning: A common wrong way

Now, to update the percentage of available threads from 10% to 20%, I changed the `<percentage>` parameter and verify it again.
```shell-script
$ echo "set_default_active_thread_percentage 20" | nvidia-cuda-mps-control
20.0
$ python3 processor_count.py
cudaDevAttrMultiProcessorCount: 4
```
Oops, It doesn't work! The number of available SMs is still 4 instead of the expected 8.

To understand the reason behind this, we need to reemphasize the details of the MPS daemon and MPS server.

When we run the command `nvidia-cuda-mps-control -d`, only an MPS control daemon is started, which is responsible for managing the startup and shutdown of the MPS server.
When an MPS client connects to the control daemon, the daemon launches an MPS server if there is no server active. This is exactly what we mentioned before when we run `python3 processor_count.py`.

The MPS control daemon does not shutdown the active server even if there are no pending client requests. This means that the active MPS server process will persist even if all active clients exit. In this case, after we run `python3 processor_count.py`, the server will remain active.

Based on the rule: the limit setting will affect all MPS servers **spawned by** the MPS control daemon after setting the limitations. Therefore, `set_default_active_thread_percentage 20` will not affect the currently active MPS server, because it was spawned after setting the active thread percentage to 10% but before updating it to 20%.

For further investigation, we manually shut down the MPS server. Then, the limit setting takes effect, and a new MPS server with a 20% active thread limit will be created, working as expected!
```shell-script
$ echo "get_server_list" | nvidia-cuda-mps-control
3448
$ echo "shutdown_server 3448" | nvidia-cuda-mps-control
$ python3 processor_count.py
cudaDevAttrMultiProcessorCount: 8

```

For the same reason, a common error occurs if we first launch the `processor_count.py` before setting the limits:
```shell-script
$ nvidia-cuda-mps-control -d                 <---- an MPS control daemon is started
$ python3 processor_count.py                 <---- an MPS server with no limits has spawned
cudaDevAttrMultiProcessorCount: 40
$ echo "set_default_active_thread_percentage 10" | nvidia-cuda-mps-control        <---- This setting will not affect the existing server
10.0
$ python3 processor_count.py
cudaDevAttrMultiProcessorCount: 40           <---- The server with no limits gives clients 100% resource access.
```
#### :white_check_mark: The correct way to update
To update the limit for an existing server, use the command `set_active_thread_percentage <PID> <percentage>`, where `<PID>` is the process ID of the running MPS server, which can be obtained using either the `nvidia-smi` command or `get_server_list` command.
```shell-script
$ nvidia-smi
Thu Mar 27 06:09:02 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:00:1E.0 Off |                    0 |
| N/A   22C    P8              10W /  70W |     31MiB / 15360MiB |      0%   E. Process |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      3530      C   nvidia-cuda-mps-server                       26MiB |
+---------------------------------------------------------------------------------------+
$ echo "get_server_list" | nvidia-cuda-mps-control
3530
$ echo "set_active_thread_percentage 3530 20" | nvidia-cuda-mps-control
20.0
$ python3 processor_count.py
cudaDevAttrMultiProcessorCount: 8
```



### Set Active Thread Limits for particular process
MPS provides finer-grain control, allowing different constraints to be set for each MPS client.
Setting `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` environment variable in an MPS client?s environment will configure the active thread percentage when the client process starts. The new limit will only further constrain the limit set by the control daemon (via `set_default_active_thread_percentage` or `set_active_thread_percentage` control daemon commands). If the control daemon has a lower setting, the control daemon setting will be obeyed by the client process instead.

We can set the environment variable inside the program. In this case, only that program will observe the limits.

```shell-script
$ cat processor_count_EnvVari.py
#!/usr/bin/env python3
# encoding: utf-8
import os
# Set the environment variable;
# this environment variable is only valid within the current process.
os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']='5'

import torch
devProp = torch.cuda.get_device_properties(torch.device('cuda'))
print("cudaDevAttrMultiProcessorCount:", devProp.multi_processor_count)

$ python3 processor_count_EnvVari.py
cudaDevAttrMultiProcessorCount: 2

$ python3 processor_count.py
cudaDevAttrMultiProcessorCount: 8

```
The result shows that only processes with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` will obey the 5% limit, while processes without the environment variable will still follow the default limits we set before.

You can also use the `export` command to set the environment variable in the terminal. In this way, any process launched from that terminal will obey the limits, as the child processes will inherit the environment from the terminal.
```shell-script
$ export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=5
$ python3 processor_count.py
cudaDevAttrMultiProcessorCount: 2
```


## Restrict process GPU memory


In CUDA 11.5, Nvidia MPS introduces a new set of control mechanisms to enable user to limit the allocation of pinned memory for MPS client processes.

### Set the default memory limit
A default global memory limit can be enabled explicitly by using the `set_default_device_pinned_mem_limit <dev> <value>` control command for the device. Setting this command enforces a device pinned memory limit on all MPS clients of all future MPS servers spawned by the MPS control daemon.

```shell-script
# shut down the active server if needed
$ echo "get_server_list" | nvidia-cuda-mps-control
2597
$ echo "shutdown_server 2597" | nvidia-cuda-mps-control

$ echo "set_default_device_pinned_mem_limit 0 2G" | nvidia-cuda-mps-control
$ python3 cuda_memory.py
memory free 2123231216 .... 2024.871094 MB
memory total 15655829504....14930.562500 MB
memory used 12905.691406 MB
```
The result shows the process available memory has been restrict from 14.58GB to 1.97 GB in GPU 0. The limitation setting works!

### Update the limit for the existing server
Similar to set the limit of active thread, to update the active server setting, we need to specify the PID of the MPS server in `set_device_pinned_mem_limit <PID> <dev> <value>` control command.

```shell-script
$ echo "get_server_list" | nvidia-cuda-mps-control
3421
$ echo "set_device_pinned_mem_limit 3421 0 4G" | nvidia-cuda-mps-control
$ python3 cuda_memory.py
memory free 4270714864 .... 4072.871094 MB
memory total 15655829504....14930.562500 MB
memory used 10857.691406 MB
```

### Set GPU memory Limits for particular process
```shell-script
$ export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=1G"
$ python3 cuda_memory.py
memory free 1049489392 .... 1000.871094 MB
memory total 15655829504....14930.562500 MB
memory used 13929.691406 MB
```



The more detail can be found in [Nvidia MPS document](https://docs.nvidia.com/deploy/mps/index.html#appendix-tools-and-interface-reference) or use command `man nvidia-cuda-mps-control`


## control interactive

Start  the  front-end management user interface to the MPS control daemon, which needs to be started first. The front-end  UI  keeps  reading
       commands  from  stdin until EOF.  Commands are separated by the newline
       character. If an invalid command is issued and rejected, an error  mes??
       sage  will be printed to stdout. The exit status of the front-end UI is
       zero if communication with the daemon is successful. A  non-zero  value
       is  returned  if the daemon is not found or connection to the daemon is
       broken unexpectedly. See the "quit" command below for more  information
       about the exit status.

```
ctl+D EOF
```

## common problem
### doesn't work, because of order
### we need to first start daemon, then set the limits

