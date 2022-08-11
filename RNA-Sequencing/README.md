# Setup steps for RNA-Sequencing

## Setting up Docker on a single Cloudlab node
Run the following commands in the shell/terminal:
1. Docker Setup.

    **a.** Uninstall all old versions.

        $ sudo apt-get remove docker docker-engine docker.io containerd runc

    **b.** Setup the repository.

        $ sudo apt-get update && sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
   
   **c.** Adding Docker's GPG key.

        $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
   
    **d.** Setting up Docker stable repository.

        $ sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    
    **e.** Installing Docker engine.

        $ sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    
    **f.** Fix permission issues and verify the installation.

        $ sudo chmod 666 /var/run/docker.sock
        
        $ docker run hello-world

2. gsutil Setup.

        $ cd /mydata && curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-326.0.0-linux-x86_64.tar.gz && tar -xvf google-cloud-sdk-326.0.0-linux-x86_64.tar.gz && /mydata/google-cloud-sdk/install.sh

    If prompted to enter a path to an rc file to update or leave blank to use, enter: /users/<username>/.bashrc.

3. GATK Workflow Setup.

    **a.** Setting up the directories for the Workflow repository.

        $ mkdir /mydata/gatk-workflows && mkdir /mydata/gatk-workflows/inputs

    **b.** Download the Cromwell jar and clone the Workflow repository.
    
        $ cd /mydata/gatk-workflows && wget https://github.com/broadinstitute/cromwell/releases/download/33.1/cromwell-33.1.jar
        
        $ cd /mydata/gatk-workflows && git clone https://github.com/Arun-George-Zachariah/gatk4-rnaseq-germline-snps-indels.git

4. Exit and restart the session.

5. Download the files required by the Workflow.

    $ gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.fasta /mydata/gatk-workflows/inputs

    $ gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.fasta.fai /mydata/gatk-workflows/inputs

    $ gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.dict /mydata/gatk-workflows/inputs

    $ gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.dbsnp138.vcf /mydata/gatk-workflows/inputs

    $ gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.dbsnp138.vcf.idx /mydata/gatk-workflows/inputs

    $ gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Mills_and_1000G_gold_standard.indels.b37.sites.vcf /mydata/gatk-workflows/inputs

    $ gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.known_indels.vcf /mydata/gatk-workflows/inputs

    $ gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Mills_and_1000G_gold_standard.indels.b37.sites.vcf.idx /mydata/gatk-workflows/inputs

    $ gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.known_indels.vcf.idx /mydata/gatk-workflows/inputs

    $ gsutil cp gs://gatk-test-data/intervals/star.gencode.v19.transcripts.patched_contigs.gtf /mydata/gatk-workflows/inputs

6. Setup EVA.
    **a.** Clone the repository.

        $ cd /mydata && git clone https://github.com/MU-Data-Science/EVA.git

    **b.** Set HOME path.

        $ cd /mydata/EVA/scripts

        $ mv /mydata/EVA/scripts/setup_tools.sh /mydata/EVA/scripts/setup_tools.sh.bkp

        $ sed '2 a HOME=/mydata' /mydata/EVA/scripts/setup_tools.sh.bkp >> /mydata/EVA/scripts/setup_tools.sh

        $ bash setup_tools.sh

7. Copy all the processing scripts to the node.

8. Setup Conda, Mamba and Snakemake.

    **a.** Create and activate conda environment.

        $ conda create -n mamba-env python=3.8

        $ conda activate mamba-env

    **b.** Inside the conda environment, install mamba.

        $ conda install -c conda-forge mamba #Failing

        $ mamba create -c conda-forge -c bioconda -n snakemake snakemake

    **c.** Export the snakemake path.

        $ echo 'export PATH=/mydata/anaconda3/envs/snakemake/bin::$PATH' >> ~/.profile && . ~/.profile

    **d.** Activate the snakemake environment inside the conda environment.

        $ conda activate mamba-env [execute again if deactivated with the above export]

        $ mamba activate snakemake

9. Setup CADD scripts.

    **a.** Clone the repository and navigate into the directory.

        $ cd /mydata && git clone https://github.com/kircherlab/CADD-scripts.git

        $ cd /mydata/CADD-scripts

    **b.** Run the following to feed the answers via script.

        $ echo "y
                y
                n
                n
                n
                y" | bash install.sh

10. Download the CADD annotations.

    $ mkdir -p /mydata/CADD-scripts/data/annotations

    $ cd /mydata/CADD-scripts/data/annotations && wget -c https://krishna.gs.washington.edu/download/CADD/v1.6/GRCh37/annotationsGRCh37_v1.6.tar.gz

    $ cd /mydata/CADD-scripts/data/annotations && gunzip annotationsGRCh37_v1.6.tar.gz && tar -xvf annotationsGRCh37_v1.6.tar

11. Copy all the scripts to their respective directory paths, mentioned in the scripts.

12. Create the directories.

    $ mkdir /mydata/InpSequences

    $ mkdir /mydata/Out_uBAM

    $ chmod +x /mydata/EVA/scripts/convert_uBAM.sh

    $ mkdir /mydata/CADD_Inp

    $ mkdir /mydata/CADD_Scores


## Running scripts for RNA-sequencing

13. Run the RNA-sequencing script.

    $ bash rna_seq.sh ID1 .. IDn

## Running scripts for CADD computation

14. Run the compute script.

    $ bash compute.sh