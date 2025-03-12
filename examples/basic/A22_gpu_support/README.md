# Support GPU Access

In this example, we demonstrate how to create hosts which can access to GPU devices.

## Create GPU-Enable Hosts
`Node::setGPUAccess` API takes parameter: whether hosts can access to GPU devices
 and it will return an `Node` instance on success. In this case, we will name our new host `gpu` and enable it access to GPU devices and connect it to the internal network net0.
 
 ```python
 gpuhost = as152.createHost('gpu').setGPUAccess(True).joinNetwork('net0')
 ```
 This provides more granular control over a GPU reservation as custom values can be set for the following device properties:
 
 * `count` specified as an integer, represents the number of GPU devices that should be reserved (providing the host holds that number of GPUs). 
 If `count` is not specified, `all` GPUs available on the host are used by default.
 
```python
 gpuhost = as152.createHost('gpu').setGPUAccess(True, count=1).joinNetwork('net0')
 ```

 * `device_ids` specified as a list of strings, represents GPU device IDs from the host. You can find the device ID in the output of `nvidia-smi` on the host. 
 If no `device_ids` are set, `all` GPUs available on the host are used by default.

```python
 gpuhost = as152.createHost('gpu').setGPUAccess(True, deviceIds=['0']).joinNetwork('net0')
  ```
You can use either `count` or `device_ids` in each of nodes. An error is returned if you try to combine both, specify an invalid device ID, or use a value of count that?s higher than the number of GPUs in your system.


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
2.4.1+cu121
True
cpu 0.16777896881103516 tensor(141257.9531)
cuda:0 0.02951216697692871 tensor(141400.6875, device='cuda:0')
cuda:0 0.0003669261932373047 tensor(141400.6875, device='cuda:0')
```
Compare the running time between `cpu` and `cuda(gpu)` case, it shows the GPU devices are accessiable.

```shell-script
$ nvidia-smi
Wed Mar 12 01:18:53 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       On  | 00000000:00:1E.0 Off |                    0 |
| N/A   26C    P8               9W /  70W |      2MiB / 15360MiB |      0%      Default |
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
