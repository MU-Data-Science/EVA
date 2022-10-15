## Changes needed to run the GATK pipeline developed for Spark/Hadoop using AVAH.

1. Assume a 16-node cluster. Using a text editor, add the below XML fragment to `/mydata/hadoop/etc/hadoop/yarn-site.xml` on `vm0`.
   ```
   <property>
                <name>yarn.nodemanager.vmem-check-enabled</name>
                <value>false</value>
   </property>
   <property>
                <name>yarn.nodemanager.pmem-check-enabled</name>
                <value>false</value>
   </property>
   ```

   If you need to change the node memory/cores allocated for YARN jobs, refer to this [page](YARN-README.md).

   Memory settings need to changed in `yarn-site.xml` as GATK executes all the variant calling tasks inside the JVM. Change the default value of `yarn.scheduler.maximum-allocation-mb` to `91440`. For `yarn.nodemanager.resource.memory-mb`, set it to say `80%` of a node's physical memory. For example, if a node has 252 GB of RAM, then set `yarn.nodemanager.resource.memory-mb` to `201000`. If JVM memory errors occur during execution, feel free to change these settings based on your genome sequences and cluster hardware.

   Execute the following commands to update `yarn-site.xml` on all the worker nodes and restart YARN:
   ```
   $ python3 ${HOME}/AVAH/scripts/run_remote_command.py copy 16 /mydata/hadoop/etc/hadoop/yarn-site.xml /mydata/hadoop/etc/hadoop/
   $ /mydata/hadoop/sbin/stop-yarn.sh
   $ /mydata/hadoop/sbin/start-yarn.sh
   ```
   If you are running a multi-site experiment with 8 nodes in Clemson (e.g., c8220, vm0-vm7) and 8 nodes in Wisconsin (e.g., c220g2, vm8-vm15), then first update the `yarn-site.xml` on all the 16 nodes using `run_remote_command.py`. Then update `yarn-site.xml` on the first 8 nodes (vm0-vm7). We will simplify this step in the near future.
2. Also edit `/mydata/spark/conf/spark-defaults.conf` on `vm0` to have the following properties:
   ```
   spark.executor.memory        24g
   spark.executor.memoryOverhead 5g
   ```
   Execute the following commands to update `spark-defaults.conf` on all the worker nodes and restart Spark:
   ```
   $ python3 ${HOME}/AVAH/scripts/run_remote_command.py copy 16 /mydata/spark/conf/spark-defaults.conf /mydata/spark/conf/
   $ /mydata/spark/sbin/stop-all.sh
   $ /mydata/spark/sbin/start-all.sh
   ```
