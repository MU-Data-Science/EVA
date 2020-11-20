#!/usr/bin/env bash

# Configurations.
CLUSTER_MACHINES="$1"
USER_NAME="$2"
PRIVATE_KEY="$3"
DATA_DIR="$4"
CONDA_SETUP_SCRIPT="conda-setup.sh"

# Iterating over the machines.
for machine in $(cat "$CLUSTER_MACHINES")
do
  # Copying the node setup script to the nodes.
  scp -i "$PRIVATE_KEY" "$CONDA_SETUP_SCRIPT" "$USER_NAME@$machine:~" &> /dev/null

  # Executing the setup script.
  ssh -o "StrictHostKeyChecking no" -i "$PRIVATE_KEY" "$USER_NAME@$machine" "~/$CONDA_SETUP_SCRIPT $DATA_DIR $" &> /dev/null &

  # Adding the pid to a list.
  bp_list="$bp_list $!"
  echo -e "\t + $machine ... OK ☕"
done

# Waiting for the setup in the nodes to finish - start.
echo -e "WAITING FOR SETUP TO FINISH \n"
total=$(cat $CLUSTER_MACHINES | wc -l | sed 's/ //')
echo $(date| tr '[:lower:]' '[:upper:]')
echo -e "CHECKING PIDS STATUS.."
finished=1
while [[ $finished -gt 0 ]]; do
	finished=$total

  states=""
	for pid in $bp_list; do
		state=$(ps -o state $pid  |tail -n +2)
		states="$states $state"
		if [[ ${#state} -eq 0 ]]; then
			finished=$((finished-1))
		fi;
	done;

  echo "REMAINING: "$finished"/"$total
	states=${states// /}
	if [[ ${#states} -gt 0 ]]; then
		sleep 30
	fi
done;

echo $(date| tr '[:lower:]' '[:upper:]')
wait
# Waiting for the setup in the nodes to finish - end.

echo -e ">> CONDA SETUP FINISHED 🌮"
exit 0