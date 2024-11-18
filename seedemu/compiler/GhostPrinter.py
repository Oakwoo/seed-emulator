from seedemu.core import Emulator, Compiler, Registry, ScopedRegistry, Node, AutonomousSystem
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
        
        info = []
        base = emulator.getLayer("Base")
        for ((scope, type, name), obj) in registry.getAll().items():
            if type == 'hnode' and obj.isGhostnode():
                self._log('compiling ghost node {} for as{}...'.format(scope, name))
                AS = base.getAutonomousSystem(obj.getAsn())
                # seperate the AS's printJson and __printJson(obj, AS) avoids endless calling.
                # because __printJson(obj) may become a function of Node class: obj.printJson() in future
                # Logically, the node informations are contained in Autonomous System information.
                # Autonomous System Class is the parent level information of Node Class.
                # i.e. AutonomousSystem.printJson has INVOKED Node.printJson
                # printJson function should only print out the children level information.
                # Invoking AS.printJson inside of Node.printJson will cause endless invoking!
                ghost_node_info = json.loads(self.__printJson(obj))
                ghost_node_info["Autonomous_System"] = json.loads(AS.printJsonBrief())
                info.append(ghost_node_info)
        
        json_str = json.dumps(info, indent=4)  
        self._log('creating ghost_nodes.json...'.format(scope, name))
        print(json_str, file=open('GhostNodes.json', 'w'))
        
