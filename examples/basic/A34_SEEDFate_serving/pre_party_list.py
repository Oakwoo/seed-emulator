import json

Fate_arch_tree = {}
Fate_arch_tree["training"] = {}
Fate_arch_tree["serving"] = {}

def createFateCluster(partyid, AutonomousSystem):
    if partyid not in Fate_arch_tree["training"]:
        Fate_arch_tree["training"][partyid] = {}
    components=["federation","proxy", "fateboard", "roll", "egg", "python", "meta-service", "mysql", "redis"]
    for component in components:
        Fate_arch_tree["training"][partyid][component]=["hnode_"+str(AutonomousSystem)+"_"+component]

def createServingFateCluster(partyid, AutonomousSystem):
    if partyid not in Fate_arch_tree["serving"]:
        Fate_arch_tree["serving"][partyid] = {}
    components=["redis", "serving-server", "serving-proxy"]
    for component in components:
        Fate_arch_tree["serving"][partyid][component]=["hnode_"+str(AutonomousSystem)+"_Serving"+component]

createFateCluster(10000, 151)
createServingFateCluster(10000, 151)
createFateCluster(9999, 152)
createServingFateCluster(9999, 152)
createFateCluster(8888, 150)
createServingFateCluster(8888, 150)

Fate_arch_tree_str = json.dumps(Fate_arch_tree, indent=4)
print(Fate_arch_tree_str, file=open('pre_party_list.json', 'w'))

