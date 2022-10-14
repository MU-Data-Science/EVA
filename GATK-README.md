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

   If you need to change the node memory/cores allocated for YARN jobs, refer to this [page](YARN-README.md). Please change the default values of memory to `91440` for `yarn.scheduler.maximum-allocation-mb` and `231440` for `yarn.nodemanager.resource.memory-mb`. This is necessary as GATK's executes all the variant calling tasks inside the JVM.

   Execute the following commands to update `yarn-site.xml` on all the worker nodes and restart YARN:
   ```
   $ python3 ${HOME}/AVAH/scripts/run_remote_command.py copy 16 /mydata/hadoop/etc/hadoop/yarn-site.xml /mydata/hadoop/etc/hadoop/
   $ /mydata/hadoop/sbin/stop-yarn.sh
   $ /mydata/hadoop/sbin/start-yarn.sh
   ```

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
