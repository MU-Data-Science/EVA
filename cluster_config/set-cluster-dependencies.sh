#!/bin/bash

cluster_addresses="$1"
username="$2"
private_key="$3"

SCALA_VER="2.11.8"

ssh_command="
# >> UPDATING REPOSITORIES AND PACKAGES..
echo 'deb https://dl.bintray.com/sbt/debian /' | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt-get update  --yes      # Fetches the list of available updates
#sudo apt-get upgrade --yes     # Strictly upgrades the current packages
#sudo apt-get dist-upgrade      # Installs updates (new ones)

# >> INSTALLING OTHERS..
sudo apt-get install default-jre --yes
sudo apt-get install default-jdk --yes
sudo apt-get install vim --yes
sudo apt-get install openssl --yes
sudo apt-get -f install
sudo apt-get install unzip --yes
sudo apt-get install software-properties-common --yes
sudo apt-get install maven --yes
sudo apt-get install jq --yes
sudo apt-get install sbt --yes

# INSTALL JAVA 8 to replace java-default
# sudo apt-get install openjdk-8-jdk --yes
# sudo apt-get install openjdk-8-jre --yes
# sudo apt-get autoremove --yes
# sudo update-alternatives --config java
# sudo update-alternatives --config javac

# echo >> INSTALLING SCALA..
sudo apt-get remove scala-library scala --yes
wget http://www.scala-lang.org/files/archive/scala-$SCALA_VER.deb
sudo dpkg -i scala-$SCALA_VER.deb
"

echo ">> NODES:"
bp_list=""
for machine in $(cat $cluster_addresses)
do
   ssh -o "StrictHostKeyChecking no" -i $private_key $username@$machine "$ssh_command" &
   bp_list="$bp_list $!"
   echo -e "\t $machine ... OK ☕"
done

echo -e "\nWAITING FOR SETUP TO FINISH..\n"
TOTAL=$(cat $cluster_addresses | wc -l | sed 's/ //')
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

echo -e ">> SETUP FINISHED 🌮"
exit 0
