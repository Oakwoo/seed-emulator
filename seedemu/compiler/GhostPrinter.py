from seedemu.core import Emulator, Compiler, Registry, ScopedRegistry, Node, Graphable
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
        
        info["Role"] = '{}'.format(node.getRole())
        
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
        for ((scope, type, name), obj) in registry.getAll().items():
            if type == 'hnode' and obj.isGhostnode():
                self._log('compiling ghost node {} for as{}...'.format(scope, name))
                info.append(json.loads(self.__printJson(obj)))
        
        json_str = json.dumps(info, indent=4)  
        self._log('creating ghost_nodes.json...'.format(scope, name))
        print(json_str, file=open('GhostNodes.json', 'w'))
        
