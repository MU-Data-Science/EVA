# Imports
import cloudlab_profile
import shlex
import subprocess
import datetime
import paramiko
import time
import constants

import geni.util as geniutil
import geni.aggregate.cloudlab as cloudlab

def create_slice(slicename, context):
    try:
        # Defining experiment validity.
        slice_exp = datetime.datetime.utcnow() + datetime.timedelta(minutes=constants.EXPERIMENT_VALIDITY_MINS)

        # Creating a slice in CloudLab.
        context.cf.createSlice(context, slicename, exp=slice_exp)
    except Exception as e:
        print "cloudlab_setup.py :: create_slice :: " + repr(e) + " - " + str(
            e) + "\tDiscarding Slice " + slicename + "."
        return 1
    else:
        return 0

def verify_setup():
    return

def create_cluster(slice_name, nodes=constants.DEFAULT_CLUSTER_SIZE, site=constants.DEFAULT_SITE):
    # Creating the context.
    context = geniutil.loadContext(key_passphrase=constants.PASS_PHRASE)

    # Creating the slice.
    result = create_slice(slice_name, context)
    if result == 1:
        print "cloudlab_setup.py :: create_cluster :: Could not create the slice :: ", slice_name
        exit(-1)

    # Creating the (standard) RSpec.
    rspec = cloudlab_profile.create_std_RSpec(nodes)

    # Obtaining the site details.
    if site == constants.SITE_UTAH:
        aggregate = cloudlab.Utah
    elif site == constants.SITE_WISCONSIN:
        aggregate = cloudlab.Wisconsin
    elif site == constants.SITE_CLEMSON:
        aggregate = cloudlab.Clemson
    elif site == constants.SITE_APT:
        aggregate = cloudlab.Apt
    else:
        print "cloudlab_setup.py :: create_cluster :: Invalid site details entered. Should be one of utah|wisc|clemson|apt"

    # Creating the experiment.
    try:
        result = aggregate.createsliver(context, slice_name, rspec)
    except Exception as excep:
        print excep
        exit(-1)

    print "cloudlab_setup.py :: create_cluster :: Created silver", result.__dict__

    # Extracting the nodes from the status.
    output = subprocess.check_output(shlex.split(
        "java -cp xml_processor/build/XMLProcessor.jar edu.missouri.core.XPathProcessor '" + aggregate.listresources(
            context, slice_name).__dict__.get("_xml") + "' '//login/@hostname'")).rstrip()
    node_lst = [x.strip() for x in output[1:-1].split(",")]
    print "cloudlab_setup.py :: create_cluster :: node_lst:: ", node_lst

    # Extracting the ports from the status.
    output = subprocess.check_output(shlex.split(
        "java -cp xml_processor/build/XMLProcessor.jar edu.missouri.core.XPathProcessor '" + aggregate.listresources(
            context, slice_name).__dict__.get("_xml") + "' '//login/@port'")).rstrip()
    port_lst = [x.strip() for x in output[1:-1].split(",")]
    print "cloudlab_setup.py :: create_cluster :: port_lst:: ", port_lst

    # Setting up ssh client.
    ssh = paramiko.SSHClient()
    sshkey = paramiko.RSAKey.from_private_key_file(constants.PRIVATE_KEY_PATH)
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Waiting for the experiment to finish setting up.
    print "cloudlab_setup.py :: create_cluster :: Awaiting completion of provisioning."
    time.sleep(500)

    # Creating a status dictionary for auditing the node status.
    status_dict = {}

    # Verifying the nodes provisioned.
    for node, port in zip(node_lst, port_lst):
        # Skipping dataset login validation.
        if port != "22":
            continue

        tries = 0
        while constants.MAX_RETRIES > tries:
            try:
                ssh.connect(hostname=node, port=22, username=constants.USER_NAME, pkey=sshkey)
                status_dict[node] = True
                break
            except Exception as e:
                time.sleep(60)
                tries += 1
                status_dict[node] = False
                print e, "\n cloudlab_setup.py :: create_cluster :: Retrying ", node, " ..."

    if not all(value for value in status_dict.values()):
        print "cloudlab_setup.py :: create_cluster :: Not all nodes were provisioned. Details :: ", status_dict
        exit(-1)
    else:
        print "cloudlab_setup.py :: create_cluster :: Successfully provisioned all nodes."

    return node_lst

def setup_cluster_config(node_lst):
    print "cloudlab_setup.py :: setup_cluster_config :: Setting up Hadoop and Spark on the nodes."

    # Executing the cluster config script. (Note: We are setting up adam presently.)
    command = 'ssh -o "StrictHostKeyChecking no" %s@%s "git clone %s && cd EVA/cluster_config && ./cluster-configure.sh %s adam"' % (constants.USER_NAME, node_lst[0], constants.GIT_URL, len(node_lst))
    process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = process.communicate(command.encode('utf-8'))
    print(out.decode('utf-8'))
    print "cloudlab_setup.py :: setup_cluster_config :: Completed setting up Hadoop and Spark on the nodes."