#!/usr/bin/env python3
# encoding: utf-8

import json
from os import mkdir, chdir, getcwd
from hashlib import md5

DockerFileTemplates: dict[str, str] = {}

DockerFileTemplates['egg'] = """\
FROM federatedai/egg:1.3.0-release
# Since CentOS 7 has reached EOL, the mirror is moved to vault
# before using yum to install， I need to run the following command
ARG repo_file=/etc/yum.repos.d/CentOS-Base.repo
RUN cp $repo_file ~/CentOS-Base.repo.backup
RUN sed -i s/#baseurl/baseurl/ $repo_file
RUN sed -i s/mirrorlist.centos.org/vault.centos.org/ $repo_file
RUN sed -i s/mirror.centos.org/vault.centos.org/ $repo_file
RUN yum clean all

# install ip command
RUN yum install iproute -y

{dockercontent}
"""

DockerFileTemplates['fateboard'] = """\
FROM federatedai/fateboard:1.3.0-release

{dockercontent}
"""

DockerFileTemplates['federation'] = """\
FROM federatedai/federation:1.3.0-release

{dockercontent}
"""

DockerFileTemplates['meta-service'] = """\
FROM federatedai/meta-service:1.3.0-release

{dockercontent}
"""

DockerFileTemplates['mysql'] = """\
FROM mysql:8
RUN microdnf install iproute
RUN microdnf install iputils

{dockercontent}
"""

DockerFileTemplates['proxy'] = """\
FROM federatedai/proxy:1.3.0-release

{dockercontent}
"""

DockerFileTemplates['python'] = """\
FROM federatedai/python:1.3.0-release
# Since CentOS 7 has reached EOL, the mirror is moved to vault
# before using yum to install， I need to run the following command
ARG repo_file=/etc/yum.repos.d/CentOS-Base.repo
RUN cp $repo_file ~/CentOS-Base.repo.backup
RUN sed -i s/#baseurl/baseurl/ $repo_file
RUN sed -i s/mirrorlist.centos.org/vault.centos.org/ $repo_file
RUN sed -i s/mirror.centos.org/vault.centos.org/ $repo_file
RUN yum clean all

# install ip command
RUN yum install iproute -y

{dockercontent}
"""

DockerFileTemplates['redis'] = """\
FROM redis:5
RUN apt-get update
RUN apt-get install iproute2 -y
RUN apt-get install iputils-ping

{dockercontent}
"""

DockerFileTemplates['roll'] = """\
FROM federatedai/roll:1.3.0-release

{dockercontent}
"""

StartScriptTemplates: dict[str, str] = {}

StartScriptTemplates['egg'] = """\
#!/bin/bash
{startCommands}

cd /data/projects/fate/eggroll/storage-service-cxx &&     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/data/projects/fate/eggroll/storage-service-cxx/third_party/lib &&     mkdir logs &&     ./storage-service -p 7778 -d /data/projects/fate/data-dir >> logs/console.log 2>>logs/error.log &     cd /data/projects/fate/eggroll/egg &&     java -cp "conf/:lib/*:eggroll-egg.jar" com.webank.ai.eggroll.framework.egg.Egg -c conf/egg.properties

tail -f /dev/null
"""

StartScriptTemplates['fateboard'] = """\
#!/bin/bash
{startCommands}

java -Dspring.config.location=/data/projects/fate/fateboard/conf/application.properties  -Dssh_config_file=/data/projects/fate/fateboard/conf  -Xmx2048m -Xms2048m -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:gc.log -XX:+HeapDumpOnOutOfMemoryError  -jar fateboard.jar

tail -f /dev/null
"""

StartScriptTemplates['federation'] = """\
#!/bin/bash
{startCommands}

java -cp "conf/:lib/*:fate-federation.jar" com.webank.ai.fate.driver.Federation -c conf/federation.properties

tail -f /dev/null
"""

StartScriptTemplates['meta-service'] = """\
#!/bin/bash
{startCommands}

java -cp "conf/:lib/*:fate-meta-service.jar" com.webank.ai.eggroll.framework.MetaService -c conf/meta-service.properties

tail -f /dev/null
"""

StartScriptTemplates['mysql'] = """\
#!/bin/bash
{startCommands}

mysqld

tail -f /dev/null
"""

StartScriptTemplates['proxy'] = """\
#!/bin/bash
{startCommands}

java -cp "conf/:lib/*:fate-proxy.jar" com.webank.ai.fate.networking.Proxy -c conf/proxy.properties

tail -f /dev/null
"""

StartScriptTemplates['python'] = """\
#!/bin/bash
{startCommands}

sleep 20; source /data/projects/python/venv/bin/activate && cd ./fate_flow &&     python fate_flow_server.py

tail -f /dev/null
"""

StartScriptTemplates['redis'] = """\
#!/bin/bash
{startCommands}

redis-server --requirepass fate_dev

tail -f /dev/null
"""

StartScriptTemplates['roll'] = """\
#!/bin/bash
{startCommands}

java -cp "conf/:lib/*:fate-roll.jar" com.webank.ai.eggroll.framework.Roll -c conf/roll.properties

tail -f /dev/null
"""

PartiesConfTemplate = """\
#!/bin/bash

user=root
dir=/data/projects/fate

partylist=({partylist})
partyiplist=({partyiplist})
servingiplist=({servingiplist})

# party 1 will host the exchange by default
exchangeip=

# flag if containers are builded by SEEDFate while compatible with original KubeFATE
build=True

buildInfo='{build_conf_json}'
"""

with open('./pre_party_list.json', 'r') as arch_json:
    arch_data = json.load(arch_json)

chdir("ghost")

with open('./AutonomousSystems.json', 'r') as as_json:
    AutonomousSystems = json.load(as_json)
    
AutonomousSystems_dic ={}
for AS in AutonomousSystems:
    AutonomousSystems_dic[AS['asn']]=AS
    
with open('./GhostNodes.json', 'r') as ghostnodes_json:
    ghostnodes = json.load(ghostnodes_json)
    
ghostnode_dic = {}
for ghostnode in ghostnodes:
    ghostnode_dic[ghostnode["NodeId"]] = ghostnode
    
# information for build docker-compose.yml
build_conf={}
# information for FATE original parties.conf 
partylist = []
partyiplist = []
servingiplist = []

for partyid, party in arch_data.items():
    node_component_dic = {}
    for component, nodeids in party.items():
        for nodeid in nodeids:
            node_component_dic[nodeid]=component
            
    # generate /etc/hosts for this party
    # go into any one component of this party
    chdir(list(party.values())[0][0])
            
    path = "/tmp/etc-hosts"
    staged_path = md5(path.encode('utf-8')).hexdigest()
    # updated etc-hosts contains all nodes information,
    # because these are other nodes which are not Fate component, 
    # should be include in ect/hosts but hard to distinguish
    with open('./'+staged_path, 'r') as input_ect_hosts:
        etc_hosts = ""
        # traversing each etc-hosts terms
        for line in input_ect_hosts:
            ip, hostname = line.replace("\n","").split(" ")
            seperate_index = hostname.find('-')
            nodeid = "hnode_"+hostname[:seperate_index]+"_"+hostname[seperate_index+1:]
            # if nodeid is in this party, add component sign
            if nodeid in node_component_dic.keys():
                etc_hosts += (line.rstrip() +" "+ node_component_dic[nodeid]+"\n")
            else:
                etc_hosts += line
    chdir("..")
    
    for component, nodeids in party.items():
        for nodeid in nodeids:
            chdir(nodeid)
            # update Dockerfile
            with open('./Dockerfile', 'r') as input_dockerfile:
                dockercontent = input_dockerfile.read()
            dockerfile = DockerFileTemplates[component].format(dockercontent = dockercontent).rstrip()
            print(dockerfile, file=open('./Dockerfile', 'w'))
            print("Generating Dockerfile for", nodeid)
        
            # update start.sh
            path = "/start.sh"
            staged_path = md5(path.encode('utf-8')).hexdigest()
            with open('./'+staged_path, 'r') as input_start_script:
                startCommands = input_start_script.read().rstrip()
            startScript = StartScriptTemplates[component].format(startCommands = startCommands).rstrip()
            print(startScript, file=open(staged_path, 'w'))
            print("Generating start.sh for", nodeid)
        
            # update /etc/hosts
            path = "/tmp/etc-hosts"
            staged_path = md5(path.encode('utf-8')).hexdigest()
            print(etc_hosts, file=open(staged_path, 'w'))
            print("Generating /etc/hosts for", nodeid)
            chdir('..')
            
    # generate parties.conf content
    build_conf[partyid] = {}
    skeleton_subnets = set()
    
    partylist.append(partyid)
    
    # TODO
    # for now only support each component corresponding to one container
    # although above component has a nodeids list 
    # in future, build_conf[partyid][component]=[{}, {}] instead of build_conf[partyid][component]={} for now
    for component, nodeids in party.items():
        nodeid = nodeids[0]
        component_build ={}
        
        current_path = getcwd()
        component_build["build_file_path"] = current_path + "/" + nodeid
        
        component_build["build_folder"] = "./" + component
        
        component_build["ip"]= ghostnode_dic[nodeid]["Interfaces"][0]["Address"]
        
        # TODO may need to discuss the detail more
        skeleton_subnet = "output_net_"+str(ghostnode_dic[nodeid]['Autonomous_Systems'])+"_"\
                          +ghostnode_dic[nodeid]["Interfaces"][0]["Connected_to"]
        component_build["skeleton_subnet"]= skeleton_subnet
        skeleton_subnets.add(skeleton_subnet)
        
        build_conf[partyid][component]=component_build
        
        if component =="proxy":
            partyiplist.append(component_build["ip"])
            servingiplist.append(component_build["ip"])
        
    build_conf[partyid]["skeleton_subnets"] = list(skeleton_subnets)

build_conf_json = json.dumps(build_conf, indent=4)
print("Generating parties.conf...")
print(PartiesConfTemplate.format(
          partylist = " ".join(partylist),
          partyiplist = " ".join(partyiplist),
          servingiplist = " ".join(servingiplist),
          build_conf_json = build_conf_json
      ), file=open("parties.conf", 'w'))



                 
    
    

    
    
    
    
