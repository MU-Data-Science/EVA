#!/usr/bin/env bash

# Input configurations.
cluster_machines="$1"
username="$2"
private_key="$3"
data_dir="$4"
share_dir="$5"
genome_tools_setup_script="genome-tools-setup.sh"

# Setup Adam.
git clone https://github.com/Arun-George-Zachariah/adam.git $data_dir/adam
echo "ADAM_HOME=$data_dir/adam" >> ~/.bashrc

# Setup Cannoli.
git clone https://github.com/Arun-George-Zachariah/cannoli.git $data_dir/cannoli
echo "CANNOLI_HOME=$data_dir/cannoli" >> ~/.bashrc

for machine in $(cat "$cluster_machines")
do
  scp -i "$private_key" "$genome_tools_setup_script" "$username@$machine:~" &> /dev/null
  ssh -o "StrictHostKeyChecking no" -i "$private_key" "$username@$machine" "~/$genome_tools_setup_script $data_dir $share_dir" &> /dev/null &
  bp_list="$bp_list $!"
  echo -e "\t + $machine ... OK ☕"
done

echo -e "WAITING FOR SETUP TO FINISH \n"
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

echo -e ">> GENOME TOOLS SETUP FINISHED 🌮"
exit 0