#!/usr/bin/env bash

nodes="$1"
user_name="$USER"
data_dir="/mydata"

if [ "$#" -ne 1 ] | [ "$#" -ne 2 ]; then
  echo "Usage: cluster-configure.sh <no. of nodes> <spark3>"
  exit -1
fi

# configuration constants.
machines="cluster-machines.txt"
shareDir="/proj/eva-public-PG0"
hadoopVer=2.7
hadoopSubVer=6
sparkVer=2.4.7

if [ $2 = spark3 ]; then
  hadoopVer=3.2
  hadoopSubVer=0
  sparkVer=3.0.0
fi

experiment=$(basename $machines)
scripts=(\
  "set-cluster-permissions" \
  "set-cluster-etchost" \
  "set-cluster-passwd" \
  "set-cluster-dependencies" \
  "set-cluster-iptables" \
  "set-cluster-hadoop" \
  "set-cluster-spark" \
  "set-cluster-conda" \
  "set-cluster-genome-tools" \
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

count=1
for script in "${scripts[@]}"
do
  log_file="LOG-"$script"-"$experiment".log"
  cmd="./$script.sh $machines $user_name ~/.ssh/id_rsa $data_dir $shareDir $hadoopVer $sparkVer $hadoopSubVer &> $log_file"
  
  eval "$cmd"
  echo ">> FINISHED ($count/${#scripts[@]}) $script.sh LOG $log_file"
  count=`expr $count + 1`
done

# Printing Hadoop report
HADOOP_STATUS_LOG="hadoop-data-nodes-alive.log"
$data_dir/hadoop/bin/hdfs dfsadmin -report | grep 'Name: ' > ${HADOOP_STATUS_LOG}
echo ">> Check ${HADOOP_STATUS_LOG} for live data nodes"

echo ">> CLEANING UP."
rm -rf $shareDir/hadoop_$install_id
rm -rf $shareDir/spark_$install_id
rm -rf $machines

echo ">> WORK IS DONE."
