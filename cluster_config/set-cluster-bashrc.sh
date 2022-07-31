#!/bin/bash
cluster_machines="$1"
username="$2"
private_key="$3"
data_dir="$4"
spark_ver="$7"

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

# If Spark 2.4.* is used w/ scala 2.12
ssh_command_spark_dist='echo "Nothing to set for SPARK_DIST_CLASSPATH"'
if [[ $spark_ver == *"2.4."* ]]; then
    ssh_command_spark_dist='echo "export SPARK_DIST_CLASSPATH=\$($HADOOP_HOME/bin/hadoop classpath)" >> ~/.bashrc'
fi

echo -e ">> READY TO DISTRIBUTE\n"

echo ">> NODES:"
for machine in $(cat "$cluster_machines")
do
    ssh -o "StrictHostKeyChecking no" -i $private_key $username@$machine "$ssh_command"
    ssh -o "StrictHostKeyChecking no" -i $private_key $username@$machine "$ssh_command_spark_dist"
    echo -e "\t + $machine ... OK ☕"
done
echo -e ">> SCRIPT FINISHED SUCCESSFULLY 🍻 \n"
