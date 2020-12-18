#!/bin/bash
INSTALL_ID="$1"
data_dir="$2"
share_dir="$3"
hadoop_ver="$4"."$5"

master_node="vm0"
node_prefix="vm"
node_start=1
node_end=$(cat /etc/hosts | grep 'vm' | wc -l)

NN_DIR="$data_dir/hadoop/hdfs/namenode"
DN_DIR="$data_dir/hadoop/hdfs/datanode"
NUMREP="3"
HSTNAME=$(hostname | cut -d '.' -f 1)
HOST_PRIVATE_IP=$(grep $HSTNAME /etc/hosts | cut -f 1)
HSTNAME_LEN=${#HSTNAME}

# Clean in case of previous install
cluster_prefix="$share_dir/hadoop_$INSTALL_ID"
rm -Rf "$cluster_prefix"
mkdir -p "$cluster_prefix"

# Verify if the hadoop distribution is available in the share directory.
if [ ! -f $share_dir/EVA_Tools/hadoop-$hadoop_ver.tar.gz ]; then
    wget https://archive.apache.org/dist/hadoop/core/hadoop-$hadoop_ver/hadoop-$hadoop_ver.tar.gz -P $share_dir/EVA_Tools/
fi

# Extract hadoop in the prefix dir
tar zxf $share_dir/EVA_Tools/hadoop-$hadoop_ver.tar.gz -C "$cluster_prefix" --strip-components 1
hadoop_prefix="$cluster_prefix"

######################
## HADOOP FILES
######################

# HADOOP-ENV
HADOOP_ENV="$hadoop_prefix/etc/hadoop/hadoop-env.sh"
echo '
export HADOOP_IDENT_STRING=$USER
export JAVA_HOME="/usr/lib/jvm/default-java"
' > $HADOOP_ENV

# CORE-SITE
CORE_SITE_FILE="$hadoop_prefix/etc/hadoop/core-site.xml"
echo '<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
  <property>
      <name>fs.defaultFS</name>
      <value>hdfs://'$master_node':9000</value>
  </property>
  <property>
      <name>hadoop.tmp.dir</name>
      <value>file:'$data_dir'/hadoop/hadoop_data/tmp</value>
  </property>
  <property>
    <name>dfs.webhdfs.enabled</name>
    <value>false</value>
  </property>
</configuration>
' > $CORE_SITE_FILE


# MAPRED-SITE
MAPRED_SITE_FILE="$hadoop_prefix/etc/hadoop/mapred-site.xml"
echo '<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>
</configuration>
' > $MAPRED_SITE_FILE


maxMemory=16384
maxvCores=16

# YARN-SITE
YARN_SITE_FILE="$hadoop_prefix/etc/hadoop/yarn-site.xml"
echo '<?xml version="1.0"?>
<configuration>
      <property>
          <name>yarn.nodemanager.aux-services</name>
          <value>mapreduce_shuffle</value>
      </property>
      <property>
          <name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
          <value>org.apache.hadoop.mapred.ShuffleHandler</value>
      </property>
      <property>
               <name>yarn.resourcemanager.resource-tracker.address</name>
               <value>'$master_node':8025</value>
      </property>
      <property>
               <name>yarn.resourcemanager.scheduler.address</name>
               <value>'$master_node':8030</value>
      </property>
      <property>
               <name>yarn.resourcemanager.address</name>
               <value>'$master_node':8050</value>
      </property>
      <property>
               <name>yarn.scheduler.maximum-allocation-mb</name>
               <value>'$maxMemory'</value>
      </property>
      <property>
               <name>yarn.scheduler.maximum-allocation-vcores</name>
               <value>'$maxvCores'</value>
      </property>

</configuration>
' > $YARN_SITE_FILE


# CAPACITY-SCHEDULER
CAPACITY_SCHEDULER_FILE="$hadoop_prefix/etc/hadoop/capacity-scheduler.xml"
echo '<?xml version="1.0"?>
<configuration>

  <property>
    <name>yarn.scheduler.capacity.maximum-applications</name>
    <value>10000</value>
    <description>
      Maximum number of applications that can be pending and running.
    </description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.maximum-am-resource-percent</name>
    <value>0.8</value>
    <description>
      Maximum percent of resources in the cluster which can be used to run
      application masters i.e. controls number of concurrent running
      applications.
    </description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.resource-calculator</name>
    <value>org.apache.hadoop.yarn.util.resource.DefaultResourceCalculator</value>
    <description>
      The ResourceCalculator implementation to be used to compare
      Resources in the scheduler.
      The default i.e. DefaultResourceCalculator only uses Memory while
      DominantResourceCalculator uses dominant-resource to compare
      multi-dimensional resources such as Memory, CPU etc.
    </description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.root.queues</name>
    <value>default,gold,silver,bronze</value>
    <description>
      The queues at the this level (root is the root queue).
    </description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.root.default.capacity</name>
    <value>25</value>
    <description>Default queue target capacity.</description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.root.gold.capacity</name>
    <value>25</value>
    <description>Gold queue target capacity.</description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.root.silver.capacity</name>
    <value>25</value>
    <description>Silver queue target capacity.</description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.root.bronze.capacity</name>
    <value>25</value>
    <description>Bronze queue target capacity.</description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.root.default.user-limit-factor</name>
    <value>1</value>
    <description>
    </description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.root.default.maximum-capacity</name>
    <value>100</value>
    <description>
    </description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.root.default.state</name>
    <value>RUNNING</value>
    <description>
    </description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.root.default.acl_submit_applications</name>
    <value>*</value>
    <description>
    </description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.root.default.acl_administer_queue</name>
    <value>*</value>
    <description>
    </description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.root.default.acl_application_max_priority</name>
    <value>*</value>
    <description>
    </description>
  </property>

   <property>
     <name>yarn.scheduler.capacity.root.default.maximum-application-lifetime
     </name>
     <value>-1</value>
     <description>
     </description>
   </property>

   <property>
     <name>yarn.scheduler.capacity.root.default.default-application-lifetime
     </name>
     <value>-1</value>
     <description>
     </description>
   </property>

  <property>
    <name>yarn.scheduler.capacity.node-locality-delay</name>
    <value>40</value>
    <description>
    </description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.rack-locality-additional-delay</name>
    <value>-1</value>
    <description>
    </description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.queue-mappings</name>
    <value></value>
    <description>
    </description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.queue-mappings-override.enable</name>
    <value>false</value>
    <description>
    </description>
  </property>

  <property>
    <name>yarn.scheduler.capacity.per-node-heartbeat.maximum-offswitch-assignments</name>
    <value>1</value>
    <description>
    </description>
  </property>


  <property>
    <name>yarn.scheduler.capacity.application.fail-fast</name>
    <value>false</value>
    <description>
    </description>
  </property>

</configuration>
' > $CAPACITY_SCHEDULER_FILE

HDFS_SITE_FILE="$hadoop_prefix/etc/hadoop/hdfs-site"
# HDFS-MASTERNODE
echo '<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
      <name>dfs.replication</name>
      <value>'$NUMREP'</value>
    </property>
    <property>
      <name>dfs.namenode.name.dir</name>
      <value>'$NN_DIR'</value>
    </property>
    <property>
      <name>dfs.permissions</name>
      <value>false</value>
    </property>
</configuration>
' > "$HDFS_SITE_FILE.masternode"


# HDFS-DATANODE
echo '<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
    <property>
      <name>dfs.replication</name>
      <value>'$NUMREP'</value>
    </property>
    <property>
          <name>dfs.datanode.data.dir</name>
          <value>'$DN_DIR'</value>
    </property>
    <property>
      <name>dfs.permissions</name>
      <value>false</value>
    </property>
</configuration>
' > "$HDFS_SITE_FILE.datanode"

SLAVES_FILE="$hadoop_prefix/etc/hadoop/slaves"
cp /dev/null $SLAVES_FILE
for node in `seq $node_start $(( ${node_end}-1 ))`
do
    echo "$node_prefix$node" >> $SLAVES_FILE
done

# Adding workers file (for newer versions of hadoop.)
cp ${SLAVES_FILE} ${hadoop_prefix}/etc/hadoop/workers
