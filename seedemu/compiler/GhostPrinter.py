from seedemu.core import Emulator, Compiler, Registry, ScopedRegistry, Node, AutonomousSystem
from seedemu.core.enums import NetworkType
import json

class GhostPrinter(Compiler):
    """!
    @brief Get all graphable object and graph them.

    """

    def getName(self) -> str:
        return 'GhostPrinter'
        
    def __printJson(self, node: Node) -> str:
        """!
        @brief print out all information about a single node in the JSON format. 
        Will create folder for node and the Json.

        @param node node to print.

        @returns node information in Json format string.
        """
        info = {}
        
        info["Name"] = node.getName()
        
        (scope, type, _) = node.getRegistryInfo()
        prefix = '{}_{}_'.format(type, scope)
        real_nodename = '{}{}'.format(prefix, node.getName())
        info["NodeId"] = real_nodename
        
        info["Role"] = '{}'.format(node.getRole())

        info["Ghost_Node"] = '{}'.format(node.isGhostnode())
        
        info["Autonomous_Systems"] = node.getAsn()
        
        info["Interfaces"] = []
        for interface in node.getInterfaces():
            info["Interfaces"].append(json.loads(interface.printJson()))
        
        info["Files"] = []
        for file in node.getFiles():
            info["Files"].append(json.loads(file.printJson()))
        
        info["Sofewares"] = list(node.getSoftware())
        
        info["Build_Commands"] = node.getBuildCommands()
        
        info["Start_Commands"] = node.getStartCommands()

        info["Post_Config_Commands"] = node.getPostConfigCommands()
        
        json_str = json.dumps(info, indent=4)
        return json_str

    def _doCompile(self, emulator: Emulator):
        registry = emulator.getRegistry()
        self._log('print ghost node information ...')
        
        node_info = []
        ASN_set = set()
        for ((scope, type, name), obj) in registry.getAll().items():
            if type == 'hnode' and obj.isGhostnode():
                self._log('compiling ghost node {} for as{}...'.format(scope, name))
                ASN_set.add(obj.getAsn())
                ghost_node_info = json.loads(self.__printJson(obj))
                node_info.append(ghost_node_info)
        
        json_str = json.dumps(node_info, indent=4)  
        self._log('creating ghost_nodes.json...')
        print(json_str, file=open('GhostNodes.json', 'w'))
        
        AS_info = []
        base = emulator.getLayer("Base")
        for asn in ASN_set:
            AS = base.getAutonomousSystem(asn)
            #TODO printJsonBrief to printJson
            as_info = json.loads(AS.printJsonBrief())
            AS_info.append(as_info)
            
        as_json_str = json.dumps(AS_info, indent=4)  
        self._log('creating Autonomous_Systems.json...')
        print(as_json_str, file=open('AutonomousSystems.json', 'w'))
        
        layers_name = [layer.getName() for layer in emulator.getLayers()]
        if 'EtcHosts' in layers_name:
            etc_hosts = emulator.getLayer('EtcHosts')
            hosts_file_content = []
            for ((scope, type, name), node) in registry.getAll().items():
                if type in ['hnode', 'snode', 'rnode', 'rs']:
                    #addresses = etc_hosts.__getAllIpAddress(node)
                    # does not have public function, maybe add later
                    addresses = []
                    for iface in node.getInterfaces():
                        address = iface.getAddress()
                        if iface.getNet().getType() == NetworkType.Bridge:
                            pass
                        if iface.getNet().getType() == NetworkType.InternetExchange:
                            pass
                        else:
                            addresses.append(address)
		    ######################################
                    for address in addresses:
                        hosts_file_content.append(f"{address} {' '.join(node.getHostNames())}")
            sorted_hosts_file_content = sorted(hosts_file_content, key=lambda x: tuple(map(int, x.split()[0].split('.'))))
            print('\n'.join(sorted_hosts_file_content), file=open('etc-hosts', 'w'))
