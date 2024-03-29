# ClusterConfig

 ClusterConfig holds the necessary setup scripts required to set up
 [Apache Spark](https://spark.apache.org),
 [Apache Hadoop](https://hadoop.apache.org), and
 [Adam/Cannoli](http://bdgenomics.org/) to run in the cluster.

## Setup
1.  Start an experiment by clicking any one of the profiles on CloudLab.

    1. [Profile1](https://www.cloudlab.us/p/EVA-public/EVA-multi-node-profile) - Homogeneous cluster in one site (e.g., Clemson or Wisc)
    2. [Profile2](https://www.cloudlab.us/p/EVA-public/EVA-singlesite-lan-profile) - Heterogeneous cluster in one site (e.g., Clemson or Wisc)
    3. [Profile3](https://www.cloudlab.us/p/EVA-public/EVA-multisite-lan-profile) - Heterogeneous cluster across two sites (e.g., Clemson and Wisc)

    The profile will ask for information such as your CloudLab user name,
    and the number of nodes required (which must be a value greater than 2)
    and a node/hardware type such
    as `c8220` (Clemson), `c240g2` (Wisc), etc. (Choose a node type that
    provides large amounts of RAM (> 128GB).) You must agree to use only
    deidentified data. The experiment will take a few minutes to start;
    so please be patient.

2. Go to your experiment and in `Topology View` click the node icon
   `vm0` and open a shell/terminal to connect to that node.
   Alternatively, you can use `SSH` to login to the node: `$ ssh -i
   /path/to/CloudLab/private_key_file
   CloudLab_username@CloudLab_hostname`. (You can also run
   [ssh-agent](https://www.ssh.com/ssh/agent) on your local machine to
   add your private key.)
3. After connecting to the master node `vm0`, clone the repository:
   ```
   $ git clone https://github.com/MU-Data-Science/EVA.git
   ```

4. To setup Hadoop, Spark, and other tools, execute the following in the
   shell/terminal. Suppose your cluster has 16 nodes.
   ```
   $ screen -S setup
   $ cd ${HOME}/EVA/cluster_config
   ```
   To use ADAM-Cannoli, do this:
   ```
   $ ./cluster-configure.sh 16 spark3
   ```
   To use GATK, do this:
   ```
   $ ./cluster-configure.sh 16 spark2
   ```
   Press "Ctrl-a" "Ctrl-d" (i.e., control-a followed by control-d) to
   detach from the screen session. To reattach, use `screen -r`.
   The configuration script can take more than 30 minutes to complete;
   so please be patient.

5. After successful cluster setup, quit the SSH session to `vm0` by typing
   `exit`. You are ready for the next steps!

## Acknowledgments
 These scripts are based on earlier versions developed by students of
 Dr. Praveen Rao, namely, [Anas Katib](https://github.com/anask),
 [Daniel Lopez](https://github.com/debarron), and
 [Hastimal Jangid](https://github.com/hastimal).
