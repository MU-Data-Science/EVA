#!/bin/bash

cluster_addresses="$1"
username="$2"
private_key="$3"

hadoop_port="50070"
yarn_port="8088"
sn_node_port="50090"

ssh_command="
sudo iptables -N Hadoop
sudo iptables -A Hadoop -j DROP
sudo iptables -I INPUT -m tcp -p tcp --dport $hadoop_port -j Hadoop
sudo iptables -N Yarn
sudo iptables -A Yarn -j DROP
sudo iptables -I INPUT -m tcp -p tcp --dport $yarn_port -j Yarn
sudo iptables -N SecNameNode
sudo iptables -A SecNameNode -j DROP
sudo iptables -I INPUT -m tcp -p tcp --dport $sn_node_port -j SecNameNode

# disable PasswordAuthentication and enable StrictHostKeyChecking
sudo sed -i -- 's/PasswordAuthentication yes/PasswordAuthentication no/g' /etc/ssh/sshd_config
sudo sed -i -- 's/StrictHostKeyChecking no/StrictHostKeyChecking yes/g'  ~/.ssh/config

#sudo service sshd restart
sudo /etc/init.d/ssh restart
"

echo ">> READY TO DISTRIBUTE THE COMMAND:"
echo -e "$ssh_command\n"

#Disabling access to Hadoop and Yarn Ports.
echo ">> NODES:"
for machine in $(cat $cluster_addresses)
do
   ssh -o "StrictHostKeyChecking no" -i $private_key $username@$machine "$ssh_command" 
   echo -e "\t + $machine ... OK ☕️"
done
echo -e ">> SCRIPT FINISHED SUCCESSFULLY 🍻 \n"

#Checking if password based authentication is diabled.
for machine in $(cat $cluster_addresses)
do
    passAuth=`ssh $username@$machine 'grep "^PasswordAuthentication" /etc/ssh/sshd_config'`
    if [[ $passAuth != *"no"* ]]; then
        echo -e "\033[0;31m Alert!!! Password Authentication is still enabled for $machine"
    fi
done

