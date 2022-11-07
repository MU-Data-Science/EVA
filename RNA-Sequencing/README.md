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

    If prompted to enter a path to an rc file to update or leave blank to use, enter: /users/username/.bashrc.

3. GATK Workflow Setup.

    **a.** Setting up the directories for the Workflow repository.

        $ mkdir /mydata/gatk-workflows && mkdir /mydata/gatk-workflows/inputs

    **b.** Download the Cromwell jar and clone the Workflow repository.
    
        $ cd /mydata/gatk-workflows && wget https://github.com/broadinstitute/cromwell/releases/download/33.1/cromwell-33.1.jar
        
        $ cd /mydata/gatk-workflows && git clone https://github.com/Arun-George-Zachariah/gatk4-rnaseq-germline-snps-indels.git

4. Exit and restart the session.

5. Download the files required by the Workflow.

    ```
    gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.fasta /mydata/gatk-workflows/inputs

    gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.fasta.fai /mydata/gatk-workflows/inputs

    gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.dict /mydata/gatk-workflows/inputs

    gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.dbsnp138.vcf /mydata/gatk-workflows/inputs

    gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.dbsnp138.vcf.idx /mydata/gatk-workflows/inputs

    gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Mills_and_1000G_gold_standard.indels.b37.sites.vcf /mydata/gatk-workflows/inputs

    gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.known_indels.vcf /mydata/gatk-workflows/inputs

    gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Mills_and_1000G_gold_standard.indels.b37.sites.vcf.idx /mydata/gatk-workflows/inputs

    gsutil cp gs://gcp-public-data--broad-references/Homo_sapiens_assembly19_1000genomes_decoy/Homo_sapiens_assembly19_1000genomes_decoy.known_indels.vcf.idx /mydata/gatk-workflows/inputs

    gsutil cp gs://gatk-test-data/intervals/star.gencode.v19.transcripts.patched_contigs.gtf /mydata/gatk-workflows/inputs
    ```

6. Setup EVA.

    **a.** Clone the repository.

        $ cd /mydata && git clone https://github.com/MU-Data-Science/EVA.git

    **b.** Set HOME path.

        $ cd /mydata/EVA/scripts

        $ mv /mydata/EVA/scripts/setup_tools.sh /mydata/EVA/scripts/setup_tools.sh.bkp

        $ sed '2 a HOME=/mydata' /mydata/EVA/scripts/setup_tools.sh.bkp >> /mydata/EVA/scripts/setup_tools.sh

        $ bash setup_tools.sh

7. Create the following directories.

    ```
    $ mkdir /mydata/InpSequences

    $ mkdir /mydata/Out_uBAM

    $ mkdir /mydata/CADD_Inp
    ```

8. Copy all the scripts/files to their respective directory paths.

    **a.** convert_uBAM.sh: /mydata/EVA/scripts/convert_uBAM.sh

    NOTE: chmod +x /mydata/EVA/scripts/convert_uBAM.sh

    **b.** recursive_convert_uBAM.sh: /mydata/EVA/scripts/recursive_convert_uBAM.sh

    NOTE: chmod +x /mydata/EVA/scripts/recursive_convert_uBAM.sh

    **c.** template.json: /mydata/gatk-workflows/gatk4-rnaseq-germline-snps-indels/template.json

    **d.** exec_gatk_wdl.sh: /mydata/gatk-workflows/exec_gatk_wdl.sh

    NOTE: chmod +x /mydata/gatk-workflows/exec_gatk_wdl.sh
    

## Running the script for RNA-sequencing

    $ bash rna_seq.sh ID1 .. IDn

# Setup steps for CADD score computation.

9. Setup Conda, Mamba and Snakemake.

    **a.** Create and activate conda environment.

        $ cd /mydata && wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

        $ bash Anaconda3-2022.05-Linux-x86_64.sh -b -p /mydata/anaconda3

        $ export PATH=/mydata/anaconda3/bin:$PATH

        $ echo 'export PATH=/mydata/anaconda3/bin:$PATH' >> ~/.profile && . ~/.profile

        $ conda init

    Repeat Step 4.

        $ conda install -c conda-forge -c bioconda snakemake mamba

10. Setup CADD scripts.

    **a.** Clone the repository and navigate into the directory.

        $ cd /mydata && git clone https://github.com/kircherlab/CADD-scripts.git

        $ cd /mydata/CADD-scripts
    
    **b.** In install.sh, go to line 215 and replace --conda-create-envs-only with --create-envs-only

    **b.** Run the following to feed the answers to install.sh.

        $ echo "y
                y
                y
                y
                n
                y" | bash install.sh

11. Download the CADD annotations. Recommended to use screen at this step.

    ```
    $ mkdir -p /mydata/CADD-scripts/data/annotations

    $ cd /mydata/CADD-scripts/data/annotations && wget -c https://krishna.gs.washington.edu/download/CADD/v1.6/GRCh37/annotationsGRCh37_v1.6.tar.gz

    $ cd /mydata/CADD-scripts/data/annotations && gunzip annotationsGRCh37_v1.6.tar.gz && tar -xvf annotationsGRCh37_v1.6.tar
    ```

12. Copy the script to its respective directory path.

    **a.** compute.sh: /mydata/CADD-scripts/compute.sh
    
    NOTE: chmod +x /mydata/CADD-scripts/compute.sh

13. Create the directory.

    ```
    $ mkdir /mydata/CADD_Scores
    ```

## Running the scripts for CADD computation

    $ bash compute.sh

<b>NOTE:</b> After installation, please check if Docker has been successfully installed. You can do so by running "docker" in your terminal.