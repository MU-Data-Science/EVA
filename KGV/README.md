# NSF-RAPID-KGV
A Knowledge Graph of Human Genome Variants for Advancing COVID-19 Research.

# Table of Contents

[Cloudlab Account](#cloudlab-account)

[Processing VCF files and CADD Scores](#processing-vcf-files-and-cadd-scores)

[Installing SnpEff](#installing-snpeff)

[Installing VCF2RDF](#installing-vcf2rdf)

[Extracting GENE ID as a Triple](#extracting-gene-id-as-a-triple)

[Processing Accession Data](#processing-accession-data)

[Team](#team)

[Acknowledgments](#acknowledgments)

# Cloudlab Account
It is recommended to use Cloudlab for generating knowledge graphs, as the dataset size is large and the computation may take longer. You may signup for a Cloudlab account [here](https://cloudlab.us/signup.php). Once you have setup an account, please follow the steps below.

Step 1: After you login, you will be able to see all the existing experiments. If you do not have one already, click on "Experiments" and then "Start experiment".

Step 2: Pick a project and select any available resource. You may check the availability by clicking on "Resource Availability".

Step 3: Choose a name for your experiment and number of nodes and click on "Next". You may change the number of hours, however, it is recommended to extend the experiment only if you require the node for longer.
[label](../Code/definitions.ttl)
Your experiment will be ready in a few minutes!

# Processing VCF files and CADD Scores
Once you have setup an account, please follow the steps below.

## Setting up Conda environment

Step 1: Set a public and private key to be able to successfully SSH into the experiment. Once successful, SSH into the experiment from your terminal for MacOS users and Putty for Windows users. The SSH command is available in the "List View", under "SSH command".

Step 2: Once you are logged in, change the directory to "/mydata" and create a new directory.

    $ cd /mydata
    $ mkdir kgsar_exp

Step 3: Clone this repository.

    $ git clone https://github.com/raopr/NSF-RAPID-KGV.git

Step 4: Create and activate a new Conda environment with Python version 3.7.3. This version is important as there are library dependencies.

    $ cd /mydata && wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
    $ bash Anaconda3-2022.05-Linux-x86_64.sh -b -p /mydata/anaconda3
    $ export PATH=/mydata/anaconda3/bin:$PATH
    $ echo 'export PATH=/mydata/anaconda3/bin:$PATH' >> ~/.profile && . ~/.profile
    $ conda init
    $ cd kgsar_exp && conda create -n name_of_environment
    $ conda activate name_of_environment

## Installing all required packages

Step 1: Install python3 and openjdk.

    $ sudo apt update
    $ sudo apt install python3-pip
    $ sudo apt-get install openjdk-11-jdk
    $ sudo apt-get install openjdk-11-jdk gradle

Step 2: Install all the required packages.

    $ pip3 install pandas
    $ pip3 install lxml [Use: sudo apt-get install python3-lxml, if this fails]
    $ pip3 install xlrd
    $ pip3 install openpyxl
    $ pip3 install rdflib
    $ pip3 install tqdm

# Installing SnpEff

Step 1: Go to home directory, '/mydata' in cloudlab and download the latest version.

    $ cd /mydata
    $ wget https://snpeff.blob.core.windows.net/versions/snpEff_latest_core.zip
    $ unzip snpEff_latest_core.zip

Step 2: Download SnpEff databases

    $ java -jar snpEff.jar download GRCh38.105

GRCh38.105 is the latest as of Dec 7, 2022.

To view all available databases: 

    $ java -jar snpEff.jar databases

To view all versions of GRCh38:

    $ java -jar snpEff.jar databases | grep GRCh38


# Installing VCF2RDF

Step 1: Install the required prerequisites.

    $ sudo apt-get update
    $ sudo apt-get install autoconf automake gcc make pkg-config zlib1g-dev guile-2.2 guile-2.2-dev gnutls-bin libraptor2-dev libhts-dev texlive curl libxml2-dev gnutls-dev

Step 2: Download the source code.

    $ curl -LO https://github.com/UMCUGenetics/sparqling-genomics/releases/download/0.99.11/sparqling-genomics-0.99.11.tar.gz
    $ tar zxvf sparqling-genomics-0.99.11.tar.gz

Step 4: Build the Sparqling-genomics tool.

    $ cd sparqling-genomics-0.99.11
    $ ./configure GUILD=/usr/bin/guild

Use the 'locate' command to find the path for the GUILD package.

Step 4: Once the configurations and installations are complete, make the following changes.

    $ cd tools/common/include
    $ vim helper.h

In the include packages section, add the following packages in this order:

    #include <gnutls/gnutls.h>
    #include <gnutls/crypto.h>

Repeat these steps for json2rdf/src/main.c, table2rdf/src/main.c, vcf2rdf/src/main.c and xml2rdf/src/main.c tools.

Step 5: Run the following commands:

    $ make

You are now ready to process VCFs and generate knowledge graphs by following the execution instructions in the [Processing-VCF-README.md](/NSF-RAPID-KGV/Processing-VCF/README.md)!

<b>NOTE:</b> The given instructions have been tested in an environment with Python version 3.7.3. If using any other Python versions, you may encounter version mismatch issues for other packages being used.

# Extracting GENE ID as a Triple

In order to group genes, we utilize the GENE_ID that is present in the ANN field after the VCF files are run through the vcf2rdf tool. To generate the triples, follow the steps in [Processing-VCF-README.md](/NSF-RAPID-KGV/Processing-VCF/README.md).

# Processing Accession Data

Accession data contains further information that we have utilized to enrich our KGs. The ontology definition is available in [defintions.ttl](/NSF-RAPID-KGV/Processing-VCF/definitions.ttl). To process these files, run as following:

    $ python3 ParseXML.py -i /input/directory/path -o /n3file/output/path


# Team
Faculty - Dr. Praveen Rao (PI, MU), Dr. Deepthi Rao (Co-PI, MU), Dr. Wesley Warren (Co-PI, MU), Dr. Peter Tonellato (Co-PI, MU), Dr. Eduardo Simoes (Co-PI, MU)

Current Ph.D. students - Shivika Prasanna

Graduated Ph.D. students - Dr. Arun Zachariah (MU, 2022)

# Acknowledgments

This work is supported by the National Science Foundation under Grant No. [2034247](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2034247&HistoricalAwards=false).
