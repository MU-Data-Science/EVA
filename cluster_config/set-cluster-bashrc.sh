#!/bin/bash
cluster_machines="$1"
username="$2"
private_key="$3"
data_dir="$4"

ssh_command='
echo "
  # Hadoop Variables
  export JAVA_HOME=/usr/lib/jvm/default-java
  export HADOOP_HOME='$data_dir'/hadoop
  export HADOOP_MAPRED_HOME=\$HADOOP_HOME
  export HADOOP_COMMON_HOME=\$HADOOP_HOME
  export HADOOP_HDFS_HOME=\$HADOOP_HOME
  export YARN_HOME=\$HADOOP_HOME
  export HADOOP_COMMON_LIB_NATIVE_DIR=\$HADOOP_HOME/lib/native
  export HADOOP_OPTS=\"-Djava.library.path=\$HADOOP_HOME/lib\"

  # Spark Variables
  export SPARK_HOME='$data_dir'/spark

  # Genome Tool Variables
  export BWA_HOME='$data_dir'/bwa
  export FREEBAYES_HOME='$data_dir'/freebayes
  export TMP_DIR='$data_dir'/tmp
  export TMPDIR='$data_dir'/tmp

  # For YARN
  export HADOOP_CONF_DIR=\$HADOOP_HOME/etc/hadoop

  # Conda Variables
  export CONDA_HOME='$data_dir'/anaconda3

  # New Path
  export PATH=\$CONDA_HOME/envs/gatk/bin:\$CONDA_HOME/bin:\$PATH:\$HADOOP_HOME/bin:\$HADOOP_HOME/sbin:\$SPARK_HOME/bin
  " >> ~/.bashrc
'

echo -e ">> READY TO DISTRIBUTE\n"

echo ">> NODES:"
for machine in $(cat "$cluster_machines")
do
    ssh -o "StrictHostKeyChecking no" -i $private_key $username@$machine "$ssh_command"
    echo -e "\t + $machine ... OK ☕"
done
echo -e ">> SCRIPT FINISHED SUCCESSFULLY 🍻 \n"
