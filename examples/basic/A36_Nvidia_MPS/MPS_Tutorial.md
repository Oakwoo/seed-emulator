# Nvidia MPS(Multi-Process Service) Tutorial
[NVIDIA Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html) is a feature that allows multiple processes to share a single GPU,
improving the utilization and efficiency of the GPU for workloads. MPS allows multiple CPU processes to launch Kernel functions simultaneously on the GPU,
combining them into a single application context for better GPU utilization.
The MPS Server manages GPU hardware resources through a CUDA Context. Multiple MPS clients send their tasks to the GPU via the MPS Server, bypassing the hardware
time-slicing scheduling limitations, which enables their CUDA Kernels to achieve true parallelism.

Due to many documents and experimental results becoming outdated with updates in NVIDIA GPU versions and architectures,
there is currently a lack of comprehensive, state-of-the-art examples on MPS.
here we provide a specific example here to validate the effectiveness of MPS.
If you got issues during validating the effectiveness, please feel free to check the `common problems` section.

## Run test application without MPS

To test the performance of MPS, we provide the test application `MPS_test.cu`. This test application just runs in a single thread on a single SM,
and simply execute a for loop adding (which will take ~1.5 seconds) before exiting and printing a message.

We first compile the test application using NVIDIA CUDA Compiler `nvcc`.
```shell-script
$ nvcc -o MPS_test MPS_test.cu
$ ./MPS_test
kernel duration: 1.427100s
```
After run `MPS_test`, it will print out the kernel duration time.

Now we launch 5 copies of our test app "simultaneously" using a bash script `mps_run` as following:
```shell-script
$ cat mps_run
#!/bin/bash
./MPS_test & ./MPS_test & ./MPS_test & ./MPS_test & ./MPS_test
$ bash mps_run
kernel duration: 8.624733s
kernel duration: 8.721921s
kernel duration: 8.733871s
kernel duration: 8.775690s
kernel duration: 8.739253s
```
As shown in the result above, considering context-switching time, we get the expected behavior that
all app instances take the expected ~`KERNEL_TIME * 5` seconds without actually enabling the MPS service,
because it does not run concurrently with an app from another process, the scheduler is switching during
kernel execution with Time-Slicing.

When the applications are running, we can find these 5 processes are running on GPU at same time using `nvidia-smi`.
Type displayed as ?C? for Compute Process.

```shell-script
$ nvidia-smi
Sat Mar 15 21:33:55 2025
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:00:1E.0 Off |                    0 |
| N/A   26C    P0              35W /  70W |    761MiB / 15360MiB |    100%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      3857      C   ./MPS_test                                  100MiB |
|    0   N/A  N/A      3858      C   ./MPS_test                                  100MiB |
|    0   N/A  N/A      3859      C   ./MPS_test                                  100MiB |
|    0   N/A  N/A      3860      C   ./MPS_test                                  100MiB |
|    0   N/A  N/A      3861      C   ./MPS_test                                  100MiB |
+---------------------------------------------------------------------------------------+
```

## Turn on MPS

This time, we start the MPS first, and repeat the test again.

To turn on the MPS, we create the folder for MPS pipes and log files
```shell-script
$ mkdir /tmp/mps_0
$ mkdir /tmp/mps_log_0
```

Then we set the environment variables.
```shell-script
$ export CUDA_VISIBLE_DEVICES=0
$ export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_0
$ export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_0
```


Next, run the following command to set the GPU's compute mode to `exclusive execution mode`
(which will be reset to `DEFAULT` after each reboot), and start the **MPS control daemon**.
```shell-script
$ sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
Set compute mode to EXCLUSIVE_PROCESS for GPU 00000000:00:1E.0.
All done.
$ nvidia-cuda-mps-control -d
```
Once the control daemon is ready, we can find the active process using the `ps` command.
```shell-script
$ ps -ef | grep mps
seed        2586       1  0 00:39 ?        00:00:00 nvidia-cuda-mps-control -d
```


## Run test application with MPS

We run the test and observe that all applications take the same amount of time to run, because the kernels are running concurrently, due to MPS.
```shell-script
$ bash mps_run
kernel duration: 1.617125s
kernel duration: 1.611048s
kernel duration: 1.635476s
kernel duration: 1.647204s
kernel duration: 1.632360s
```

we see that both apps take the same amount of time to run ( ~ `KERNEL_TIME`), because the kernels are running concurrently, due to MPS.


we check the kernel information about these 5 processes using `nvidia-smi` again.
We can find the running MPS server, which is displayed as type "C" for Compute Process,
and 5 running ./MPS_test processes where the type "M+C" indicates that all the processes are being scheduled by MPS.
```shell-script
$ nvidia-smi
Sat Mar 15 21:36:13 2025
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:00:1E.0 Off |                    0 |
| N/A   27C    P0              36W /  70W |   1417MiB / 15360MiB |    100%   E. Process |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      3887    M+C   ./MPS_test                                   98MiB |
|    0   N/A  N/A      3888    M+C   ./MPS_test                                   98MiB |
|    0   N/A  N/A      3889    M+C   ./MPS_test                                   98MiB |
|    0   N/A  N/A      3890    M+C   ./MPS_test                                   98MiB |
|    0   N/A  N/A      3891    M+C   ./MPS_test                                   98MiB |
|    0   N/A  N/A      3892      C   nvidia-cuda-mps-server                       26MiB |
+---------------------------------------------------------------------------------------+

```

So far, by comparing the running times without and with MPS, we have verified that MPS enables multiple GPU applications to run concurrently on a single GPU.

## Turn off MPS

To shut down the MPS control daemon, use the following commands. The daemon will shut down the servers before exiting:
```shell-script
$ echo quit | nvidia-cuda-mps-control
$ sudo nvidia-smi -i 0 -c DEFAULT
```
The system targets the corresponding MPS control daemon instance by looking up the PID stored
in `nvidia-cuda-mps-control.pid` file, located in the CUDA_MPS_PIPE_DIRECTORY folder.
Thus, you must use the terminal where MPS environment variables are set to shut it down, otherwise, you may receive the error like following:
```shell-script
$ echo quit | nvidia-cuda-mps-control
Cannot find MPS control daemon process
```
Note, the command does not forcibly turn off the MPS. If the server is still running, you can shut it down by using `sudo kill <PID>` to terminate the service.


## Common Problems
Here, we list several common problems we encountered while learning MPS. We hope these can provide some guidance for you.

* ### `nvcc` command not found
When you try to compile the test application using NVIDIA CUDA Compiler `nvcc`, you may got the error as following:
```
$ nvcc -o MPS_test MPS_test.cu
Command 'nvcc' not found, but can be installed with:
apt install nvidia-cuda-toolkit
Please ask your administrator.
```
This error occurs because the system cannot find the executable file for `nvcc`.
If you have installed the correct CUDA version, system will be able to find `nvcc` once the environment configuration is set up properly.
```shell-script
$ sudo vi ~/.bashrc
```
Append the following two lines at the end of the `~/.bashrc` file.
```
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```
 The `cuda` is a symbolic link that points to the actual **runtime api** CUDA version installed on your host machine. You can find the **runtime api** CUDA version
by navigating to the `/usr/local/` folder to locate the corresponding folder name.

 For example, in my case, my **runtime api** CUDA version is `12.1`, I can check it like this:
 ```
 $ ll /usr/local/cuda
 lrwxrwxrwx 1 root root 21 May 21  2024 /usr/local/cuda -> /usr/local/cuda-12.1/
 ```
 Finally, save the changes, make the configuration file valid, and check whether `nvcc` is set up correctly.
 ```shell-script
 $ source ~/.bashrc
 $ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
 ```
You may notice that I introduced the term **runtime API** CUDA version. The version printed out by `nvcc --version` (which is 12.1)
differs from the version shown by `nvidia-smi` in previous (which is 12.2). This discrepancy occurs because `nvcc` refers to the **runtime API** CUDA version,
while `nvidia-smi` displays the version of the CUDA **driver API** that is linked to the installed NVIDIA driver. You may have used a separate GPU driver installer
to install the GPU driver, which can result in a mismatch between the versions displayed by `nvidia-smi` and `nvcc --version`.

 The **driver API** is more low-level and less user-friendly, as it requires explicit device initialization, for example. On the other hand, the **runtime API** wraps
the driver API, automating many steps that would otherwise require manual coding (such as device initialization). This makes the runtime API much easier to use
for most users. Typically, the driver API version is backward compatible with the runtime API version. This means that the version shown by `nvidia-smi` (driver API version) being
higher than the version shown by `nvcc --version` (runtime API version) usually doesn?t cause major issues.

* ### Driver/library version mismatch
When trying to resolve the `nvcc command not found` issue, be cautious when using the suggestion `sudo apt install nvidia-cuda-toolkit`, as it may install an incorrect version of CUDA.
This can lead to errors when running the `nvidia-smi` command.

 ```shell-script
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
$ nvidia-smi
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 570.86
```
The CUDA Toolkit Installer typically includes the GPU driver installer. If you install everything through the CUDA Toolkit Installer, the runtime API and driver API versions should be consistent.
The above error information may be caused by installing two different versions of the driver by different approaches.
In this case, CUDA driver version 10.1 mismatch with the nvidia module NVML 570.86. One is installed by CUDA Toolkit Installer(10.1 version) and the other is installed by separate GPU driver(12.2 version).

 To solve this problem, we only need to execute the following command.
 ```shell-script
 sudo apt-get --purge remove "*nvidia*"
 ```
 Then the problem boils down to common problem #1. You can follow the provided solution to resolve it.

* ### Segmentation fault
A segmentation fault typically occurs when multiple processes are running on the GPU while the GPU's compute mode is set to `exclusive execution mode`.
  * When you launch multiple test applications **without** MPS using the bash script `mps_run`, if you got segmentation fault error like following:
```shell-script
$ bash mps_run
mps_run: line 3:  3067 Segmentation fault      (core dumped) ./MPS_test
```
This issue occurs because the GPU's compute mode is accidentally set to `exclusive execution mode`. Under this mode, GPU allows only a single process can run on the GPU at a time.
Thus, once a test applications is running on the GPU, the rest applications will be terminate because CUDA-capable device(s) is/are busy.

   To solve the issue, we simply need to change the mode to `DEFAULT`. This will allow multiple processes to share the GPU resources. You can do this using the following command:
```shell-script
$ nvidia-smi -i 0 -c DEFAULT
```
 * Even **with** the MPS server running, launching bash script without setting the environment variables ahead will result in the following error:
```shell-script
$ bash mps_run
mps_run: line 2:  3383 Segmentation fault      (core dumped) ./MPS_test
mps_run: line 2:  3384 Segmentation fault      (core dumped) ./MPS_test
mps_run: line 2:  3385 Segmentation fault      (core dumped) ./MPS_test
```
The reason is the same: without setting these variables, the program will run independently, conflicting with the `exclusive execution mode`.
To solve this, you need to set the environment variables.
```shell-script
$ export CUDA_VISIBLE_DEVICES=0
$ export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_0
$ export CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_0
```


* ### CUDA-capable device(s) is/are busy or unavailable
During the test application run **with** MPS, if you launch multiple test applications from **multiple terminals**, it is essential to set the environment variables in **each** terminal.
Without setting these variables, the program will run independently and will not be scheduled under`nvidia-cuda-mps-server`, which is responsible for merging them into a single process for concurrent execution.
Because we have set the GPU to `exclusive execution mode` to enable the MPS server, failing to set the environment variables will lead to the following error:
```shell-script
$ ./MPS_test
RuntimeError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

* ### Try this experiment using Tensorflow
If you want to use Tensorflow to test the performance of MPS.
Currect tensorflow can't support cuda 12. The error is when I try to import
tensorflow, I got error like following:
```
2025-03-14 01:38:54.168815: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-14 01:38:54.171037: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
2025-03-14 01:38:54.220461: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
2025-03-14 01:38:54.221044: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-03-14 01:38:55.122922: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
```
It still doens't work after I install TensorFlowRT using `pip3 install nvidia-tensorrt` or `pip3 install tensorflow[and-cuda]`
based on [@arivero's answer](https://stackoverflow.com/questions/75614728/cuda-12-tf-nightly-2-12-could-not-find-cuda-drivers-on-your-machine-gpu-will),
The main reason is that as of March 2023, the only tensorflow distribution for cuda 12 is the docker package from NVIDIA.
The solutoin should be install docker with the nvidia cloud instructions and run one of the recent containers

 Thus, I go to [Nvidia's offcial webset about Tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow).
I find a lasted AMD64 version docker image, and run following commands:
```
docker run --gpus all -it --rm nvcr.io/nvidia/tensorflow:25.02-tf2-py3
```
Now Tensorflow can see the GPUs.
```
python
Python 3.12.3 (main, Jan 17 2025, 18:03:48) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
2025-03-14 02:10:32.274568: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-03-14 02:10:32.291991: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2025-03-14 02:10:32.313317: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8473] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2025-03-14 02:10:32.320570: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1471] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-03-14 02:10:32.335969: I tensorflow/core/platform/cpu_feature_guard.cc:211] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
>>> tf.config.list_physical_devices("GPU").__len__() > 0
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1741918239.268167     150 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
I0000 00:00:1741918239.349616     150 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
I0000 00:00:1741918239.353080     150 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
True
>>> tf.sysconfig.get_build_info()
OrderedDict({'cpu_compiler': '/opt/rh/gcc-toolset-11/root/usr/bin/gcc', 'cuda_compute_capabilities': ['sm_100', 'sm_120', 'sm_75', 'sm_80', 'sm_86', 'compute_90'], 'cuda_version': '12.8', 'cudnn_version': '9', 'is_cuda_build': True, 'is_rocm_build': False, 'is_tensorrt_build': True})
>>>
```

* ### GPU behavior changes due to the evolution of GPU architectures
As the Nvidia GPU architectures has developed, the scheduler behavior in the **without-MPS** case when running kernels from multiple processes appears to have changed.
Here we briefly conclude the process of CPU changes and explain the reasons behind them.
 * **On pre-Pascal GPUs** (e.g. Kepler architectures), the test applications provided by [Robert Crovella](https://stackoverflow.com/questions/34709749/how-do-i-use-nvidia-multi-process-service-mps-to-run-multiple-non-mpi-cuda-app/34711344#34711344)
 simply waits for a period of time (~5 seconds) before exiting. The expected behavior is as follows:
 ```shell-script
 $ ./mps_run
 kernel duration: 6.409399s
 kernel duration: 12.078304s
 ```
 one instance of the app takes the expected ~5 seconds, whereas the other instance takes approximately double that ( ~10 seconds)
 because since it does not run concurrently with an app from another process, it waits for 5 seconds while the other app/kernel is running, and then spends 5 seconds running its own kernel, for a total of of ~10 seconds.

 * **On Pascal or newer GPUs**, the  "time-sliced" scheduler is switching to another kernel from another process during kernel execution, rather than wait for a kernel from one process to complete.
 As observed by [lewisreg](https://forums.developer.nvidia.com/t/cuda-kernels-from-different-process-can-run-concurrently-same-performance-with-mps-on-and-off/60789),
 the test applications provided by Robert Crovella behaves differently. In this case, all the processes nearly start at the same time due to the scheduler switching between processes.
 Unlike CUDA computing, which allows only one process per clock cycle, waiting can occur concurrently. As a result, all kernels finish their waiting time at almost the same moment, causing them to each last approximately 5 seconds.
 It creates the illusion that different processes are running concurrently.
 ```
 $ bash wait_5sec
 kernel duration: 4.030812s
 kernel duration: 4.033008s
 kernel duration: 4.034832s
 kernel duration: 4.045161s
 kernel duration: 4.050006s
 ```
  Thus, a test application using a simple wait for a period of time no longer works once the GPU applies "time-sliced" scheduler,
  as we can't distinguish the concurrence by comparing the running time with or without MPS.

   As mentioned in lewisreg's comments, change the time limit condition to the loop condition `(loop_count<MAX_COUNT)` will yield the expected behavior.
  When verifying this, we found that the empty loops were "optimized out" by the compiler, resulting in a very short running time (~ 0.17 seconds).

   We can use the `nvprof` command to check the details. It shows that the GPU kernel
  itself takes only 1.18 �s, while the 98% of the time is spent on `cudaLaunchKernel` during the preparation stage.
  ```
  $ nvprof ./empty_loop
==4202== NVPROF is profiling process 4202, command: ./empty_loop
kernel duration: 0.333002s
==4202== Profiling application: ./empty_loop
==4202== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  1.1840us         1  1.1840us  1.1840us  1.1840us  delay_kernel(unsigned int)
      API calls:   98.01%  143.52ms         1  143.52ms  143.52ms  143.52ms  cudaLaunchKernel
                    1.97%  2.8835ms       114  25.293us     151ns  1.5796ms  cuDeviceGetAttribute
                    0.01%  11.144us         1  11.144us  11.144us  11.144us  cuDeviceGetName
                    0.00%  6.2030us         1  6.2030us  6.2030us  6.2030us  cuDeviceGetPCIBusId
                    0.00%  3.9760us         1  3.9760us  3.9760us  3.9760us  cudaDeviceSynchronize
                    0.00%  2.3680us         3     789ns     265ns  1.7970us  cuDeviceGetCount
                    0.00%     772ns         2     386ns     170ns     602ns  cuDeviceGet
                    0.00%     571ns         1     571ns     571ns     571ns  cudaGetLastError
                    0.00%     465ns         1     465ns     465ns     465ns  cuDeviceTotalMem
                    0.00%     374ns         1     374ns     374ns     374ns  cuModuleGetLoadingMode
  ```
  Running 5 app instances simultaneously without MPS causes the app instances to take 5 times longer (~ 0.84 seconds) than running a single process.
  However, running with MPS introduces extra time to set context compared to directly sending tasks to the GPU.
  This overhead becomes critical when the running time is extremely short, making the total running
  time with MPS several times longer than running a single process(~ 0.57 seconds).

  Therefore, we improved the test application by replacing the empty loop with a simple addition operation, which makes the running time reasonably longer
  to mitigate the extra time noise introduced by MPS.

   The testing results on the Nvidia Tesla T4 GPU on AWS is shown as below table (pre-Pascal GPUs data is from Robert Crovella's comments).
    We have provided the codes in `./wait`, `./empty_loop` folders. Please feel free to test them on your own devices.
 <div style="text-align:center">
   <table>
	<tr>
	    <th colspan="7" style="text-align:center" >Running time of app instances on the Nvidia Tesla T4 GPU on AWS</th>
	</tr >
	<tr>
	    <td rowspan="2"> </td>
	    <td colspan="2">without MPS</td>
	    <td rowspan="2">with MPS</td>
      <td rowspan="2">single</td>
      <td rowspan="2">is related to # of processes?</td>
      <td rowspan="2">does MPS improve performance?</td>
	</tr >
	<tr >
      <td>pre-Pascal GPUs</td>
	    <td>Pascal or newer GPUs</td>
	</tr>
	<tr>
	    <td>wait</td>
      <td>6.41s 12.07s</td>
	    <td>4.03s 4.03s 4.03s 4.05s 4.05s</td>
       <td>3.94s 3.96s 3.97s 3.97s 3.98s</td>
       <td>3.32s</td>
       <td>:x:</td>
       <td>:x:</td>
	</tr>
  <tr>
	    <td>empty loop</td>
      <td>NaN</td>
	    <td>0.84s 0.85s 0.87s 0.97s 0.97s</td>
       <td>0.58s 0.59s 0.59s 0.60s 0.61s</td>
       <td>0.18s</td>
       <td>:white_check_mark:</td>
       <td>:warning: not obvious</td>
	</tr>
  <tr>
	    <td>simple adding</td>
      <td>NaN</td>
	    <td>8.62s 8.72s 8.73s 8.78s 8.74s</td>
       <td>1.62s 1.61s 1.64s 1.65s 1.63s</td>
       <td>1.43s</td>
       <td>:white_check_mark:</td>
       <td>:white_check_mark:</td>
	</tr>

</table>
</div>

 #### Time slicing vs MPS
 Using time-slicing, although the scheduler switch between processes, it still doesn't mean that kernels from separate processes are running "concurrently".
 In each clock cycles, there is only **one** kernel is executing on GPU, i.e.  the code from kernel A is not executing in the same clock cycle(s) as the code from kernel B,
 when A and B originate from separate processes in the non-MPS case. For CUDA programs that are not computation-intensive, most of the CUDA cores remain idle.
 While MPS fully utilizes the GPU's CUDA cores, with each program running on different CUDA cores during every clock cycle.

 **Time slicing:** Unlike waiting, which can happen concurrently, if we run `N` processes simultaneously, only `1/N` of the clock cycles are allocated to each process.
 Thus, the running time with time-slicing (without MPS) is **`N` times longer** than running a single process.

 **MPS:** As long as the CUDA cores are sufficient to run all processes, the running time with MPS is the **same as** running a single process.









<!--
# this part I am not 100% sure, just have hypothesis.
Mention that in code, we add one line sleep code as following:
``` python
def train(net, trainloader, optimizer, epochs, device):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        time.sleep(10)
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    return net
```
No matter time.sleep(10) exists or not, we didn't observe the expected behavior on CNN program.
The reason may be
1) pytorch spreads CNN training work on all SM, in this case, even turn on MPS, MPS can't schedule two processes on GPU,
because all SM has been occupied by one process.
(to verify, maybe try to restrict CNN train on one SM)
2) CNN is not all running on CUDA, it has some work, such as loading data, is finished by CPU.
When CPU is running, GPU is idle when we run single process.
In multiple processes case, when CPU runs process A, GPU runs process B, then CPU runs process B, GPU runs process A. looks like they are running concurrently.
-->

