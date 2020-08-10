#!/usr/bin/env bash

nodes="$1"
user_name="$USER"
data_dir="/mydata"

if [ "$#" -ne 1 ]; then
  echo "Usage: cluster-configure.sh <no. of nodes>"
  exit -1
fi

# configuration constants.
machines="cluster-machines.txt"
shareDir="/proj/eva-public-PG0"

experiment=$(basename $machines)
scripts=(\
  "set-cluster-permissions" \
  "set-cluster-etchost" \
  "set-cluster-passwd" \
  "set-cluster-dependencies" \
  "set-cluster-iptables" \
  "set-cluster-hadoop" \
  "set-cluster-spark" \
  "set-cluster-adam" \
  "set-cluster-bashrc")

# Write the node list to cluster-machines.txt.
for ((i=0;i<$nodes;i++)); do
  # Generating the node name.
  nodeName="vm"$i

  # Writing to a file.
  echo $nodeName >> $machines
done

# Generating a private key on the master node.
ssh-keygen -q -t rsa -N '' -f /users/$USER/.ssh/id_rsa <<<y 2>&1 >/dev/null

# Copying the contents of the public key to authorized keys across hosts.
pubKey=$(cat /users/$USER/.ssh/id_rsa.pub)
for machine in $(cat $machines)
do
  sudo ssh -o "StrictHostKeyChecking no" $machine "echo $pubKey >> /users/$USER/.ssh/authorized_keys"
done

# Generate a file to set a unique variable
install_id=$(date +"%s")
echo "$install_id" > "$machines-INSTALL_ID.txt"

echo ">> WAIT FOR IT YOUNG BLOOD 👽 ID: $install_id"
for script in "${scripts[@]}"
do
  log_file="LOG-"$script"-"$experiment".log"
  cmd="./$script.sh $machines $user_name ~/.ssh/id_rsa $data_dir $shareDir&> $log_file"
  
  eval "$cmd"
  echo ">> FINISHED $script.sh LOG $log_file 🕶"
done

echo "Cleaning up."
rm -rvf $shareDir/hadoop_$install_id
rm -rvf $shareDir/spark_$install_id

echo ">> WORK IS DONE 🥃"
