#!/usr/bin/env python3
# encoding: utf-8

from seedemu.layers import Base, Routing, Ebgp, EtcHosts
from seedemu.services import WebService
from seedemu.compiler import *
from seedemu.core import Emulator, Binding, Filter
import os
from seedemu import *

def run(dumpfile = None):
    # Initialize the emulator and layers
    emu     = Emulator()
    base    = Base()
    routing = Routing()
    ebgp    = Ebgp()
    web     = WebService()
    etc_hosts = EtcHosts()

    ###############################################################################
    # Create an Internet Exchange
    base.createInternetExchange(100)

    ###############################################################################
    # Create and set up AS-150

    # Create an autonomous system 
    as150 = base.createAutonomousSystem(150)

    # Create a network 
    as150.createNetwork('net0')

    # Create a router and connect it to two networks
    as150.createRouter('router0').joinNetwork('net0').joinNetwork('ix100')

    # Create a host called web and connect it to a network
    as150.createHost('web').joinNetwork('net0')

    # Create a web service on virtual node, give it a name
    # This will install the web service on this virtual node
    web.install('web150')

    # Bind the virtual node to a physical node 
    emu.addBinding(Binding('web150', filter = Filter(nodeName = 'web', asn = 150)))


    ###############################################################################
    # Create and set up AS-151
    # It is similar to what is done to AS-150

    as151 = base.createAutonomousSystem(151)
    as151.createNetwork('net0')
    as151.createRouter('router0').joinNetwork('net0').joinNetwork('ix100')

    as151.createHost('web').joinNetwork('net0')
    web.install('web151')
    emu.addBinding(Binding('web151', filter = Filter(nodeName = 'web', asn = 151)))

    ###############################################################################
    # Create and set up AS-152
    # It is similar to what is done to AS-150
    access_key = "f2930a4a168240f995a1a21c79603bf2"

    as152 = base.createAutonomousSystem(152)
    as152.createNetwork('net0')
    as152.createRouter('router0').joinNetwork('net0').joinNetwork('ix100')
    
    as152.createHost('web').joinNetwork('net0')
    web.install('web152')
    emu.addBinding(Binding('web152', filter = Filter(nodeName = 'web', asn = 152)))
    # add host name, so flower-supernode can find flower-superlink by name
    gpuhost = as152.createHost('gpu').setGPUAccess(True).joinNetwork('net0').addHostName('flower-superlink')
    gpuhost.addSoftware('wget')
    # have to use conda to upgrade python to 3.10, because python3.8 can't pip install flwr[simulation]>=1.15.2
    gpuhost.addBuildCommand("export CONDA_DIR=/opt/conda && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p $CONDA_DIR && rm /tmp/miniconda.sh && export PATH=$CONDA_DIR/bin:$PATH && conda init && conda create -y -n py310 python=3.10")
    gpuhost.importFile(hostpath = os.path.dirname(os.path.realpath(__file__)) + "/quickstart-pytorch", containerpath = "/quickstart-pytorch")
    # in Non-interactive shell, e.g. /bin/bash -c "source ~/.bashrc" does't work
    # even conda init has modified ~/.bashrc file.
    # we have to manually load the specific conda.sh
    # but "conda init" is still needed for further interactive command

    # TODO file import is after fedprox， I can't install environment in build command
    # for now, my solution is changed the compiler Docker.py order.
    gpuhost.addBuildCommand("/bin/bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate py310 && pip3 install -e /quickstart-pytorch'")

    gpuhost.appendStartCommand(f"source /opt/conda/etc/profile.d/conda.sh && conda activate py310 && export LD_LIBRARY_PATH=$HOME/miniconda3/envs/fedprox/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH && flower-superlink --insecure", fork=True, isPostConfigCommand=True)

    num_edge_devices = 5
    gpuhosts_list = {}
    for i in range(num_edge_devices):
        gpuhosts_list[i] = as152.createHost(f'gpu{i}').setGPUAccess(True).joinNetwork('net0')
        gpuhosts_list[i].addSoftware('wget')
        gpuhosts_list[i].addBuildCommand("export CONDA_DIR=/opt/conda && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && bash /tmp/miniconda.sh -b -p $CONDA_DIR && rm /tmp/miniconda.sh && export PATH=$CONDA_DIR/bin:$PATH && conda init && conda create -y -n py310 python=3.10")
        gpuhosts_list[i].importFile(hostpath = os.path.dirname(os.path.realpath(__file__)) + "/quickstart-pytorch", containerpath = "/quickstart-pytorch")
        gpuhosts_list[i].addBuildCommand("/bin/bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate py310 && pip3 install -e /quickstart-pytorch'")
        gpuhosts_list[i].appendStartCommand(f"source /opt/conda/etc/profile.d/conda.sh && conda activate py310 && export LD_LIBRARY_PATH=$HOME/miniconda3/envs/fedprox/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH && flower-supernode      --insecure      --superlink flower-superlink:9092      --clientappio-api-address 127.0.0.1:9094      --node-config 'partition-id={i} num-partitions={num_edge_devices}'", fork=True, isPostConfigCommand=True)

    ###############################################################################
    # Create hybrid AS.
    # AS99999 is the emulator's autonomous system that routes the traffics 
    #   to the real-world internet
    as99999 = base.createAutonomousSystem(99999)
    as99999.createRealWorldRouter('rw-real-world', prefixes=['0.0.0.0/1', '128.0.0.0/1']).joinNetwork('ix100', '10.100.0.99')
    
    ###############################################################################
    # Peering these ASes at Internet Exchange IX-100

    ebgp.addRsPeer(100, 150)
    ebgp.addRsPeer(100, 151)
    ebgp.addRsPeer(100, 152)
    ebgp.addPrivatePeerings(100, [152],  [99999], PeerRelationship.Provider)


    ###############################################################################
    # Rendering 

    emu.addLayer(base)
    emu.addLayer(routing)
    emu.addLayer(ebgp)
    emu.addLayer(web)
    emu.addLayer(etc_hosts)

    if dumpfile is not None:
        emu.dump(dumpfile)
    else:
        emu.render()

        imageName = 'nvidia/cuda:12.3.1-base-ubuntu20.04'
        image  = DockerImage(name=imageName, local=False, software=[])
        docker = Docker()
        docker.addImage(image)
        docker.setImageOverride(gpuhost, imageName)
        for i in range(num_edge_devices):
            docker.setImageOverride(gpuhosts_list[i], imageName)

        ###############################################################################
        # Compilation
        emu.compile(docker, './output', override=True)

if __name__ == '__main__':
    run()
