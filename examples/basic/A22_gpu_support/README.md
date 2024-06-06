# Support GPU Access

In this example, we demonstrate how to create hosts which can access to GPU devices.

## Create GPU-Enable Hosts
The `AutonomousSystem::createHost` API takes two parameters, `name`,
which is the name of the host, `gpuAccess`, whether hosts can access to GPU devices
 and it will return an `Node` instance on success. In this case, we will name our new host `gpu` and enable it access to GPU devices and connect it to the internal network net0.
 
 ```python
 gpuhost = as152.createHost('gpu2').setGPUAccess(True).joinNetwork('net0')
 ```


## Use custom image on GPU-Enable Hosts

The custom docker image `nvidia/cuda:12.3.1-base-ubuntu20.04` from the Docker Hub is needed.

```python
imageName = 'nvidia/cuda:12.3.1-base-ubuntu20.04'
image  = DockerImage(name=imageName, local=False, software=[])
docker = Docker()
docker.addImage(image)
docker.setImageOverride(gpuhost, imageName)
```

## Run Testing cases
We provide two testing programs `GPUTest.py` and `GPUTest_Time.py` in `gpuhost` root directory.
```shell-script
$ python3 GPUTest.py 
CUDA available
Drive:  cuda:0
GPU Version:  Tesla T4
```
It shows the GPU information including Driver version and GPU model.
```shell-script
$ python3 GPUTest_Time.py 
2.3.0+cu121
True
cpu 0.1663663387298584 tensor(141185.7500)
cuda:0 0.027685165405273438 tensor(141329.3906, device='cuda:0')
cuda:0 0.0003292560577392578 tensor(141329.3906, device='cuda:0')

```
Compare the running time between `cpu` and `cuda(gpu)` case, it shows the GPU devices are accessiable.

```shell-script
$ nvidia-smi
Wed Jun  5 16:57:00 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:00:1E.0 Off |                    0 |
| N/A   37C    P8               9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+

```
You can run `nvidia-smi` to get more detail information about GPU devices.
