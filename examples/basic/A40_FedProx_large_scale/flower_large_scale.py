#!/usr/bin/env python3
# encoding: utf-8

from seedemu.compiler import Docker, Platform
from seedemu.layers import Base, Routing, Ebgp, EtcHosts
from seedemu.core import Emulator
from examples.internet.B03_hybrid_internet import hybrid_internet
import os, sys
from seedemu import *

###############################################################################
# Set the platform information
script_name = os.path.basename(__file__)

if len(sys.argv) == 1:
    platform = Platform.AMD64
elif len(sys.argv) == 2:
    if sys.argv[1].lower() == 'amd':
        platform = Platform.AMD64
    elif sys.argv[1].lower() == 'arm':
        platform = Platform.ARM64
    else:
        print(f"Usage:  {script_name} amd|arm")
        sys.exit(1)
else:
    print(f"Usage:  {script_name} amd|arm")
    sys.exit(1)

# usually in B00_mini_internet run ./mini_internet.py. output folder will be generated
# In following way, we treat it as an class and provide the dumpfile path
hybrid_internet.run('hybrid_internet.bin', 5)

emu = Emulator()
emu.load('hybrid_internet.bin')

etc_hosts = EtcHosts()

flowerhosts_list = []
base: Base = emu.getLayer('Base')
# first statistic how many flower supernodes
num_edge_devices = 0
for asn in base.getAsns():
    as_system = base.getAutonomousSystem(asn)
    num_edge_devices += len(as_system.getHosts())
print("The number of flower supder nodes:", num_edge_devices)
# set flower supernodes: enable all hosts GPU access and conda env
i = 0
for asn in base.getAsns():
    as_system = base.getAutonomousSystem(asn)
    for host_name in as_system.getHosts():
        host = as_system.getHost(host_name)
        host.setGPUAccess(True)
        host.appendStartCommand(f"source /opt/conda/etc/profile.d/conda.sh && conda activate py310 && export LD_LIBRARY_PATH=$HOME/miniconda3/envs/fedprox/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH && flower-supernode      --insecure      --superlink flower-superlink:9092      --clientappio-api-address 127.0.0.1:9094      --node-config 'partition-id={i} num-partitions={num_edge_devices}'", fork=True, isPostConfigCommand=True)
        i += 1
        flowerhosts_list.append(host)

# set flower superlink
as152 = base.getAutonomousSystem(152)
superlink = as152.createHost('gpu').setGPUAccess(True).joinNetwork('net0').addHostName('flower-superlink')
superlink.appendStartCommand(f"source /opt/conda/etc/profile.d/conda.sh && conda activate py310 && export LD_LIBRARY_PATH=$HOME/miniconda3/envs/fedprox/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH && flower-superlink --insecure", fork=True, isPostConfigCommand=True)
flowerhosts_list.append(superlink)

# Add the etc_hosts layer
emu.addLayer(etc_hosts)

# Render the emulation and further customization
emu.render()

imageName = 'fedprox_base'
dirName = './fedprox_base'
image  = DockerImage(name=imageName, dirName=dirName ,local=True, software=[])
docker = Docker()
docker.addImage(image)
for flowerhost  in flowerhosts_list:
    docker.setImageOverride(flowerhost, imageName)
emu.compile(docker, './output', override=True)

# Copy the image folder to the output folder
os.system('cp -r '+ dirName  + ' output/')
