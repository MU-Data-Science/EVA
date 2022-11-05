#!/bin/bash

MACHINES_FILE="$1"
USR="$2"
KEY="$3"
data_dir="$4"

ssh_command="
sudo chown -R '$USR' '$data_dir'
sudo mkdir '$data_dir'/var
sudo mv /var/* '$data_dir'/var
sudo mv /var /var.old
sudo ln -s '$data_dir'/var /var
"

echo ">> READY TO DISTRIBUTE COMMAND:"
echo -e "$ssh_command\n"

echo ">> NODES:"
for machine in $(cat $MACHINES_FILE) 
do
  ssh -o "StrictHostKeyChecking no" -i "$KEY"  $USR@$machine "$ssh_command"
  echo -e "\t + $machine ... OK ☕️"
done

echo -e ">> SCRIPT FINISHED SUCCESSFULLY 🍻"
