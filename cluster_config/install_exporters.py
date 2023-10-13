# Run this script to install and run the relevant
# exporters in each node of a cluster
# "cluster_configure.sh" MUST BE RUN BEFORE THIS SCRIPT

import subprocess
import sys

# Ensure that argument count is correct
if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} <no. of hosts>")
    exit(1)

# Credentials to log into the vms
num_hosts = int(sys.argv[1])

# Scripts to execute (may need to add more as more metrics are needed)
cmds = ['./install_node_exporter.sh', './run_gpu_exporter.sh']

# Execute a command as a subprocess
def execute_command(ssh_cmd):
    try:
        output = subprocess.call(ssh_cmd, shell=True)
    except subprocess.CalledProcessError as e:
        output = e.stderr
    
    return output

# Run each command on each worker vm
for host in range(1, num_hosts):
    for cmd in cmds:
        ssh_command = f"ssh -o StrictHostKeyChecking=no vm{host} 'bash -s' < {cmd}" # Runs the script on a machine that does not have a copy of the script
        print(ssh_command)
        execute_command(ssh_command)