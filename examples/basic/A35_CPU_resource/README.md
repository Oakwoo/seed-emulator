# Support Resource (CPU / Memory) limitation and allocation
By default, hosts in emulator have no resource limits and will use as much of the machine's resources as possible. If no resource limits are set for hosts, they may interfere 
with each other. Hosts that consume more hardware resources can end up consuming all available resources, causing other hosts to have no resources left and leading to service downtime.

In this example, we demonstrate how to specify the hosts' hardware resource (CPU / Memory).

## Create CPU / Memory constrained Hosts
The `Node::setCPUResource()` API takes two optional parameters, `reservation`,
which is the CPU requirement of this host, `limit`, which is the limitation of CPU of this host.
It will return an `Node` instance on success. 

The `Node::setMemoryResource()` API takes two optional parameters, `reservation`,
which is the memory requirement of this host, `limit`, which is the limitation of memory of this host.
The value must be in the form of an integer followed by a qualifier, either 'G' or 'M' that specifies the value in Gigabyte or Megabyte respectively.
It will return an `Node` instance on success. 
 
 ```python
 as151.createHost('web').setCPUResource(reservation=0.5, limit=0.6).setMemoryResource(reservation='50M', limit='500M').joinNetwork('net0')
 ```
In the above example, we have create a host `web` which requires at least `50%` of the CPUs and `50MB` memory while CPU is limited to `60%` of the available CPUs 
and memory usage is limited to a maximum of `500MB` at same time.

This ensures that no matter how resource-intensive the processes running inside the container are, they will not exceed this set limits.

## Check if the limits are effective.
The `docker inspect` command allows you to view detailed configuration information of a container, including resource limits.

First, we need to find the ID or name of the running container. You can use the `docker ps` command to get this information.
For example, if we're looking for the container we just create, we can run the following command:
```shell-script
$ dockps
0d51aff8cb6d  as100rs-ix100-10.100.0.100
b5124b43526d  as150h-web-10.150.0.71
6aba718c0274  as150r-router0-10.150.0.254
c5756b53df92  as151h-web-10.151.0.71
45d78a474822  as151r-router0-10.151.0.254
ec30af25877f  as152h-web-10.152.0.71
8e92123afee4  as152r-router0-10.152.0.254
c91c5cc5daf5  seedemu_internet_map
```
Based on the result, we can get the container ID is `c5756b53df92`. Thus, we can run the following command:
```shell-script
$ docker inspect c5756b53df92
...
"HostConfig": {
            ...
            "Memory": 524288000,
            "NanoCpus": 600000000,
            ...
            "MemoryReservation": 52428800,
            ...
        },
...
```
We see the `NanoCpus` parameter, which represents the CPU time that the container can use, in nanoseconds. 
For example, we have limited the container's CPU usage to 0.6, thus we see the `NanoCpus` value to 600000000 (which is 0.6 CPU cores in nanoseconds).
We also see the `MemoryReservation` is set to `52428800 bytes = 50 MB` and `memory` (limites of memory) is set to `524288000 bytes = 500 MB`

It shows our setting are effective!

Because the essence of Docker implementing this limits is through modifying `cgroups` (control groups), a feature of the Linux kernel, to limit, control, and isolate 
a group of processes (such as CPU, memory etc.). we can also log into the container to check if the limits are effective, using following commands 
```shell-script
$ docker exec -it c5756b53df92 /bin/bash
root@c5756b53df92 / # cat /sys/fs/cgroup/memory/memory.limit_in_bytes
524288000
root@c5756b53df92 / # cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 
100000
root@c5756b53df92 / # cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
60000
```
We can see that the container has been successfully limited to `524288000 bytes = 500MB` of memory. 

`--cpu-period` sets the scheduling period for each container
 process. It defines how often the CPU scheduler should check and enforce CPU limits for the container.
`--cpu-quota` sets the amount of CPU time that the container is allowed to use within each period. It defines the maximum CPU time the container can consume during a single scheduling period.
Together, these two options help control and limit the CPU usage of a container by specifying the period and quota of CPU time it can use.

In this case, we confirm we have limited the container's CPU usage to `60000/100000 = 60%`.
