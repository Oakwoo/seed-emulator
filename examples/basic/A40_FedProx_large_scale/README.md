In this example, we demostarte how to build a large scale fedprox using pre-build internet.

1. create federated learning edge devices, i.e. flower super nodes
The pre-build hybrid internet has 12 stub Autonomous Systems. Thus, specifing `hosts_per_as` controls the scale.

In the following example, we create 60 edge devices by setting `hosts_per_as = 5`
```
hybrid_internet.run('hybrid_internet.bin', 5)
```

2. create federated learning server, i.e. flower super link
```
as152 = base.getAutonomousSystem(152)
superlink = as152.createHost('gpu').setGPUAccess(True).joinNetwork('net0').addHostName('flower-superlink')
superlink.appendStartCommand(f"source /opt/conda/etc/profile.d/conda.sh && conda activate py310 && export LD_LIBRARY_PATH=$HOME/miniconda3/envs/fedprox/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH && flower-superlink --insecure", fork=True, isPostConfigCommand=True)
flowerhosts_list.append(superlink)
```

3. change the config file in task folder `./fedprox/pyproject.toml`
```
[tool.flwr.app.config.algorithm]
...
min-available-clients= 60
min-fit-clients= 60
...
num-clients = 60

[tool.flwr.federations.local-deployment]
...
options.num-supernodes = 60
```
