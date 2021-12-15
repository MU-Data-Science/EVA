#!/bin/bash
cluster_machines="$1"
username="$2"
private_key="$3"
data_dir="$4"
share_dir="$5"
hadoop_ver="$6"
spark_ver="$7"

# 1 Copy spark-setupsh to master node and execute it
spark_script="spark-setup.sh"
install_id=$(head -1 "$cluster_machines-INSTALL_ID.txt")

echo -e ">> EXECUTING: $spark_script\n"
master_node=$(head -1 "$cluster_machines")
scp -i "$private_key" "$spark_script" "$username@$master_node:~" &> /dev/null
ssh -i "$private_key" "$username@$master_node" "~/$spark_script $install_id $data_dir $share_dir $spark_ver $hadoop_ver" &> /dev/null

# 2 Configure nodes
echo -e ">> CONFIGURING NODES 🤖\n"
ssh_command="
if [ ! -d $share_dir/spark_$install_id ]; then
    echo 'Copying Spark files'
    scp -r vm0:$share_dir/spark_$install_id $data_dir/spark
else
    sudo cp -r $share_dir/spark_$install_id $data_dir/spark
fi

sudo chown -R $username $data_dir/spark

host_private_name=\$(hostname | cut -d '.' -f 1)
host_private_ip=\$(grep \"\\<\$host_private_name\\>$\" /etc/hosts | cut -f 1)
echo \"export SPARK_LOCAL_IP=\$host_private_ip\" >> $data_dir/spark/conf/spark-env.sh
"

bp_list=""
for machine in $(cat "$cluster_machines")
do
  ssh -o "StrictHostKeyChecking no" -i "$private_key" "$username@$machine" "$ssh_command" &> /dev/null &
  bp_list="$bp_list $!"
  echo -e "\t + $machine ... OK ☕"
done

echo -e "\nWAITING FOR SETUP TO FINISH..\n"
TOTAL=$(cat $cluster_machines | wc -l | sed 's/ //')
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

echo "Starting Spark."
$data_dir/spark/sbin/start-all.sh

echo "Creating /spark-events directory."
$data_dir/hadoop/bin/hdfs dfs -mkdir /spark-events

echo -e ">> SETUP FINISHED 🌮"
exit 0



