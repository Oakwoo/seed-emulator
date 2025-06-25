# Support GPU Resource limitation
By default, hosts in emulator have no resource limits and will use the whole GPU devices resource. With the rise of powerful GPUs like NVIDIA A100 and A800, simulating large-scale federated learning on a single machine has become feasible. However, edge devices often have limited GPU capabilities. Allowing each simulated node to fully utilize the GPU leads to unrealistic performance assumptions. To better reflect real-world scenarios, it is essential to limit GPU resources per node, enabling more accurate modeling of device heterogeneity and performance.

In this example, we demonstrate how to specify the hosts' GPU resource.

## Create GPU constrained Hosts
The `Node::setGPUAccess()` API takes two optional parameters, `activeThread`,
which is the `{activeThread}%` of GPU streaming processor resource can be used by this node, `memoryLimit`, which is the limitation of GPU memory of this host.
It will return an `Node` instance on success.


 ```python
 as152.createHost('gpu').setGPUAccess(gpuAccess=True, count=1, activeThread=5, memoryLimit='0=1G').joinNetwork('net0')
 ```
In the above example, we created a host `gpu` with only `5%` of the GPU's streaming multiprocessors, which equals `2` SMs (since the NVIDIA T4 has 40 SMs), and limited it to `1 GB` of GPU memory.

## Check if the limits are effective.
First, we need to find the ID or name of the running container using the `dockps` command. Then, we can use `docksh` to log into the container we just created and run the test program. Note that the only reason we import executable file `processor_count` and `cuda_memory` instead of their python code is because python code depends on heavy library `torch`.

```shell-script
$ chmod +x processor_count
$ ./processor_count
cudaDevAttrMultiProcessorCount: 2

$ chmod +x cuda_memory
$ ./cuda_memory
memory free 1049489392 .... 1000.871094 MB
memory total 15655829504....14930.562500 MB
memory used 13929.691406 MB
```
It shows our setting are effective!
