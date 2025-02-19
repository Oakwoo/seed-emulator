import json
Fate_arch_tree = {}
node_component = {}
def createFateCluster(partyid, AutonomousSystem):
    Fate_arch_tree[partyid] = {}
    components=["federation","proxy", "fateboard", "roll", "egg", "python", "meta-service", "mysql", "redis"]
    for component in components:
        Fate_arch_tree[partyid][component]=["hnode_"+str(AutonomousSystem)+"_"+component]
        node_component["hnode_"+str(AutonomousSystem)+"_"+component]=component
createFateCluster(10000, 151)
createFateCluster(9999, 152)
createFateCluster(8888, 150)
Fate_arch_tree_str = json.dumps(Fate_arch_tree, indent=4)
node_component_str = json.dumps(node_component, indent=4)
print(Fate_arch_tree_str, file=open('pre_party_list.json', 'w'))
print(node_component_str, file=open('node_component.json', 'w'))
