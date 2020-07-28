#!/usr/bin/env bash

# Input configurations.
cluster_machines="$1"
username="$2"
private_key="$3"
data_dir="$4"
share_dir="$5"
genome_tools_setup_script="genome-tools-setup.sh"

# Setting Java Home.
export JAVA_HOME=/usr/lib/jvm/default-java

# Installing maven
MAVEN_RELEASE="apache-maven-3.6.3-bin.tar.gz"
wget https://downloads.apache.org/maven/maven-3/3.6.3/binaries/${MAVEN_RELEASE} -O ${data_dir}/${MAVEN_RELEASE}
tar xvfz ${data_dir}/${MAVEN_RELEASE} -C ${data_dir}
rm -rf ${data_dir}/${MAVEN_RELEASE}
export M2_HOME=${data_dir}/apache-maven-3.6.3
export PATH=$M2_HOME/bin:$PATH

echo "👉 Done with installing Maven 😎"

for machine in $(cat "$cluster_machines")
do
  scp -i "$private_key" "$genome_tools_setup_script" "$username@$machine:~" &> /dev/null
  ssh -o "StrictHostKeyChecking no" -i "$private_key" "$username@$machine" "~/$genome_tools_setup_script $data_dir" &> /dev/null &
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

# Get Adam and install
git clone https://github.com/bigdatagenomics/adam.git $data_dir/adam
cd $data_dir/adam
./scripts/move_to_scala_2.12.sh
./scripts/move_to_spark_3.sh
mvn install

echo "👉 Done with installing Adam 😎"

# get Cannoli and install
git clone https://github.com/bigdatagenomics/cannoli.git $data_dir/cannoli
cd $data_dir/cannoli
./scripts/move_to_scala_2.12.sh
./scripts/move_to_spark_3.sh
mvn install
cd

echo "👉 Done with installing Cannoli 😎"