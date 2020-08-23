import constants

import geni.portal as portal
import geni.rspec.pg as pg

def create_std_RSpec(num_nodes = 4):
    print "cloudlab_profile.py :: create_std_RSpec :: Start"

    # Creating a Request object to start building the RSpec.
    request = portal.context.makeRequestRSpec()

    # Initializing a list to hold the node details.
    node_lst = []

    # Dynamically creating the nodes.
    for i in xrange(num_nodes):
        # Adding a raw PC to the request
        node = request.RawPC("vm%d" % i)

        # Specifying the default disk image.
        node.disk_image = constants.UBUNTU_16_DISK_IMAGE

        # Creating Block Storage.
        bs = node.Blockstore("bs%d" % i, constants.BLOCKSTORE_DIRECTORY)
        bs.size = constants.BLOCKSTORE_SIZE

        # Changing the blockstore permissions.
        bs_perm_cmd = "sudo chmod 777 /mydata"
        node.addService(pg.Execute(shell="bash", command=bs_perm_cmd))

        node_lst.append(node)

    # Creating a link between the nodes.
    request.Link(members=node_lst)

    return request