<details>
    <summary> Dataset Creation for Training</summary>

    1. Setting up Conda environment

        cd /mydata && wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
        bash Anaconda3-2022.05-Linux-x86_64.sh -b -p /mydata/anaconda3
        export PATH=/mydata/anaconda3/bin:$PATH
        echo 'export PATH=/mydata/anaconda3/bin:$PATH' >> ~/.profile && . ~/.profile
        conda init
        exit

    2. SSH back into the experiment.

    3. Installing required packages

        cd /mydata && conda create -n dglconda
        conda activate dglconda
        pip3 install torch
        conda install -c dglteam dgl
        pip3 install pandas matplotlib tqdm requests pymantic pyarrow
        pip3 install --upgrade psutil
        sudo add-apt-repository ppa:ubuntu-toolchain-r/test
        sudo apt-get update
        sudo apt-get install gcc-4.9
        sudo apt-get install --only-upgrade libstdc++6
        sudo apt install openjdk-11-jre-headless
        sudo apt install openjdk-11-jre-headless gradle

    4. Setting up Blazegraph

        mkdir /mydata/dgl && cd dgl
        wget https://github.com/blazegraph/database/releases/download/BLAZEGRAPH_2_1_6_RC/blazegraph.jar

    5. Create dataset

        1. Load all files to Blazegraph:

        Open a screen session:

            screen -S screen1
            ./run.sh <path to data>

        2. Run code to create the dataset for distributed DGL:

                python3 dataset_creation.py -e /path/to/dump/edges -n /path/to/dump/nodes -d /path/to/dump.pkl


    Note: Load all the files to BG because it doesn't come with an inbuilt journal file -- it creates a new one. Query as normal.

</details>

<details>
    <summary> Distributed Training </summary>

    1. Setting up Conda environment
    
        cd /mydata && wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
        bash Anaconda3-2022.05-Linux-x86_64.sh -b -p /mydata/anaconda3
        export PATH=/mydata/anaconda3/bin:$PATH
        echo 'export PATH=/mydata/anaconda3/bin:$PATH' >> ~/.profile && . ~/.profile
        conda init
        exit

    2. SSH back into the node.

    3. Installing DGL

        pip install --pre dgl -f https://data.dgl.ai/wheels/repo.html
        pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
        pip3 install torch pandas matplotlib tqdm requests
        pip3 install --upgrade psutil
        sudo add-apt-repository ppa:ubuntu-toolchain-r/test
        sudo apt-get update
        sudo apt-get install gcc-4.9
        sudo apt-get install --only-upgrade libstdc++6

    4. Set Python path

        sudo rm /usr/bin/python3
        sudo ln -s /mydata/anaconda3/bin/python3 /usr/bin/python3

    # Setting up NFS for Distributed DGL

        1. Downloading and installing components

            Host:

                sudo apt update
                sudo apt install nfs-kernel-server

            Client:

                sudo apt update
                sudo apt install nfs-common

        2. Creating the share directory on the Host's /mydata folder

            Host:

                sudo mkdir /mydata/dgl/general

            Check permissions:

                ls -la /mydata/dgl/general

        NFS will translate any root operations on the client to the nobody:nogroup credentials as a security measure. Therefore, you need to change the directory ownership to match those credentials.

            sudo chown nobody:nogroup /mydata/dgl/general

        3. Configuring the NFS Exports on the Host Server

            Host:

                sudo nano /etc/exports

            Inside /etc/exports:

                /mydata/dgl/general client1.IP.address(rw,sync,no_subtree_check) client2.IP.address(rw,sync,no_subtree_check) client3.IP.address(rw,sync,no_subtree_check)

        4. Restart Host server

            sudo systemctl restart nfs-kernel-server

        5. Adjusting the Firewall on the Host

            sudo ufw status

        6. Creating Mount Points and Mounting Directories on the Client

            Client:

                sudo mkdir -p /mydata/dgl/general
                sudo mount host.IP.address:/mydata/dgl/general /dgl/general

            Check disk usage:

                df -h

        7. Unmounting

            sudo umount /mydata/dgl/general

        8. Setting SSH permissions between Server and Client: Do the following on the Server and then copy the key to "Manage SHH keys" section of Cloudlab.

            ssh-keygen -t rsa -b 4096 -C "spn8y@umsystem.edu"
            cat ~/.ssh/id_rsa.pub 

            vim ~/.ssh/config
                Host clnode171.clemson.cloudlab.us
                Hostname 10.10.1.1
                User Shivikap
                PubKeyAuthentication yes
                IdentityFile ~/.ssh/id_rsa 

            ssh-copy-id -i ~/.ssh/id_rsa.pub -p 22 Shivikap@client.IP.address

        9. Adjust Python Path on Server and Clients

            sudo rm /usr/bin/python3
            sudo ln -s /mydata/anaconda3/bin/python3.9 /usr/bin/python3
            sudo ln -s /usr/share/pyshared/lsb_release.py /mydata/anaconda3/lib/python3.9/site-packages/lsb_release.py

</details>

# Sources: 

[1] https://docs.dgl.ai/tutorials/blitz/1_introduction.html#sphx-glr-tutorials-blitz-1-introduction-py

[2] https://www.digitalocean.com/community/tutorials/how-to-set-up-an-nfs-mount-on-ubuntu-18-04

[3] https://www.redhat.com/sysadmin/nfs-server-client

[4] https://docs.dgl.ai/tutorials/dist/1_node_classification.html#sphx-glr-download-tutorials-dist-1-node-classification-py

# Troubleshooting sources:

[1] https://stackoverflow.com/questions/44967202/pip-is-showing-error-lsb-release-a-returned-non-zero-exit-status-1

[2] https://askubuntu.com/questions/1406192/lsb-release-error-bash-usr-bin-lsb-release-usr-bin-python3-bad-interpreter
