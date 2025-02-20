#!/usr/bin/env python3
# encoding: utf-8

from seedemu.layers import Base, Routing, Ebgp, EtcHosts
from seedemu.services import WebService
from seedemu.compiler import Docker, GhostPrinter
from seedemu.core import Emulator, Binding, Filter
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

    components=["federation","proxy", "fateboard", "roll", "egg", "python", "meta-service", "mysql", "redis"]
    for component in components:
        as150.createGhostHost(component).joinNetwork('net0')

    serving_components=["redis", "serving-server", "serving-proxy"]
    for component in serving_components:
        as150.createGhostHost("Serving"+component).joinNetwork('net0')

    # Create a host called web and connect it to a network
    #as150.createHost('web').joinNetwork('net0')

    # Create a web service on virtual node, give it a name
    # This will install the web service on this virtual node
    #web.install('web150')

    # Bind the virtual node to a physical node 
    #emu.addBinding(Binding('web150', filter = Filter(nodeName = 'web', asn = 150)))


    ###############################################################################
    # Create and set up AS-151
    # It is similar to what is done to AS-150

    as151 = base.createAutonomousSystem(151)
    as151.createNetwork('net0')
    as151.createRouter('router0').joinNetwork('net0').joinNetwork('ix100')

    components=["federation","proxy", "fateboard", "roll", "egg", "python", "meta-service", "mysql", "redis"]
    for component in components:
        as151.createGhostHost(component).joinNetwork('net0')

    serving_components=["redis", "serving-server", "serving-proxy"]
    for component in serving_components:
        as151.createGhostHost("Serving"+component).joinNetwork('net0')

    #web.install('web151')
    #emu.addBinding(Binding('web151', filter = Filter(nodeName = 'web', asn = 151)))

    ###############################################################################
    # Create and set up AS-152
    # It is similar to what is done to AS-150

    as152 = base.createAutonomousSystem(152)
    as152.createNetwork('net0')
    as152.createRouter('router0').joinNetwork('net0').joinNetwork('ix100')

    components=["federation","proxy", "fateboard", "roll", "egg", "python", "meta-service", "mysql", "redis"]
    for component in components:
        as152.createGhostHost(component).joinNetwork('net0')

    serving_components=["redis", "serving-server", "serving-proxy"]
    for component in serving_components:
        as152.createGhostHost("Serving"+component).joinNetwork('net0')

    #web.install('web152')
    #emu.addBinding(Binding('web152', filter = Filter(nodeName = 'web', asn = 152)))

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

        ###############################################################################
        # Compilation
        emu.compile(Docker(), './output', override=True)
        emu.compile(GhostPrinter(), './ghost', override=True)

if __name__ == '__main__':
    run()
