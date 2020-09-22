#!/bin/bash
cluster_machines="$1"
username="$2"
private_key="$3"
data_dir="$4"
share_dir=$5
hadoop_ver=$6

master_node=$(head -1 "$cluster_machines")
data_nodes=$(tail -n+2 "$cluster_machines")

hadoop_script="hadoop-setup.sh"
install_id=$(head -1 "$cluster_machines-INSTALL_ID.txt")

# 1 Copy hadoop-setup.sh to master node and execute it
echo -e ">> EXECUTING: $hadoop_script\n"
master_node=$(head -1 "$cluster_machines")
scp -o "StrictHostKeyChecking no" -i "$private_key" "$hadoop_script" "$username@$master_node:~" &> /dev/null
ssh -o "StrictHostKeyChecking no" -i "$private_key" "$username@$master_node" "~/$hadoop_script $install_id $data_dir $share_dir $hadoop_ver && hostname | cut -d'.' -f1 > ~/masters" &> /dev/null

# 2 Configure masternode
echo -e ">> CONFIGURING MASTER NODE 😎\n"
ssh_command="
sudo cp -r $share_dir/hadoop_$install_id $data_dir/hadoop
sudo chown -R $username $data_dir/hadoop
mv ~/masters $data_dir/hadoop/etc/hadoop/

rm $data_dir/hadoop/etc/hadoop/hdfs-site.xml
rm $data_dir/hadoop/etc/hadoop/hdfs-site.datanode
mv $data_dir/hadoop/etc/hadoop/hdfs-site.masternode $data_dir/hadoop/etc/hadoop/hdfs-site.xml

sed  -i -e 's/<value>0.1<\/value>/<value>0.8<\/value>/' $data_dir/hadoop/etc/hadoop/capacity-scheduler.xml
"
ssh -o "StrictHostKeyChecking no" -i "$private_key" "$username@$master_node" "$ssh_command" &> /dev/null


# 3 Configuring datanodes
echo -e ">> CONFIGURING DATA NODES 🤓"
ssh_command="
sudo cp -r $share_dir/hadoop_$install_id $data_dir/hadoop
sudo chown -R $username $data_dir/hadoop

rm $data_dir/hadoop/etc/hadoop/hdfs-site.xml
rm $data_dir/hadoop/etc/hadoop/hdfs-site.masternode
mv $data_dir/hadoop/etc/hadoop/hdfs-site.datanode $data_dir/hadoop/etc/hadoop/hdfs-site.xml
sed  -i -e 's/<value>0.1<\/value>/<value>0.8<\/value>/' $data_dir/hadoop/etc/hadoop/capacity-scheduler.xml
"
echo ">> READY TO DISTRIBUTE COMMAND: "
echo -e "$ssh_command \n"

echo ">> NODES:"
bp_list=""
for machine in $data_nodes
do
    ssh -o "StrictHostKeyChecking no" -i $private_key $username@$machine "$ssh_command" &
    bp_list="$bp_list $!"
    echo -e "\t + $machine ... OK ☕"
done
echo -e ">> SCRIPT FINISHED SUCCESSFULLY 🍻 \n"

echo -e "\nWAITING FOR SETUP TO FINISH..\n"

TOTAL=$(echo "$data_nodes" | wc -l | sed 's/ //g')

DATE=$(date| tr '[:lower:]' '[:upper:]')
echo $DATE
echo -e "CHECKING PIDS STATUS.."
FINISHED=1
while [[ $FINISHED -gt 0 ]]; do
	FINISHED=$TOTAL
	
  states=""
	for pid in $bp_list; do 
		state=$(ps -o state $pid  |tail -n +2)
		states="$states $state"
		if [[ ${#state} -eq 0 ]]; then
			FINISHED=$((FINISHED-1))
		fi;
	done;
	
  #echo $states
  echo "REMAINING: "$FINISHED"/"$TOTAL
	states=${states// /}
	if [[ ${#states} -gt 0 ]]; then
		sleep 30
	fi
done;

DATE=$(date| tr '[:lower:]' '[:upper:]')
echo $DATE
wait

# Starting hadoop
echo "Starting up hadoop."
$data_dir/hadoop/bin/hdfs namenode -format
$data_dir/hadoop/sbin/start-dfs.sh
$data_dir/hadoop/sbin/start-yarn.sh

echo -e ">> SETUP FINISHED 🌮"
exit 0
