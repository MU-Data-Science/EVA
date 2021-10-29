## How to change YARN settings before executing AVAH

1. Connect to `vm0` using SSH.
2. Open /mydata/hadoop/etc/hadoop/yarn-site.xml using a text editor.
3. For Clemson (c8220) machines, set
   `yarn.nodemanager.resource.memory-mb` to `191440`.
4. For Wisconsin (c220g2/c220g5), set
   `yarn.nodemanager.resource.memory-mb` to `111440`.
5. You can also change `yarn.scheduler.maximum-allocation-vcores` to 72
and `yarn.scheduler.minimum-allocation-vcores` to 1.
6. Run the following command to copy `yarn-site.xml` to all worker nodes (assuming a 16-node cluster).
```
$ python3 ${HOME}/AVAH/scripts/run_remote_command.py copy 16 /mydata/hadoop/etc/hadoop/yarn-site.xml /mydata/hadoop/etc/hadoop/
```
7. Restart YARN.
```
$ /mydata/hadoop/sbin/stop-yarn.sh
$ /mydata/hadoop/sbin/start-yarn.sh
```

8. Check to see if the new configuration is in effect.
```
$ yarn node -list -all -showDetails
```
