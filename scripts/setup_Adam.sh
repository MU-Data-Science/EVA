#!/usr/bin/env bash

#PROJECT_NAME=/proj/nosql-json-PG0
DATA_DIR=/mydata

# install java jdk jre
#sudo apt-get update
#sudo apt-get install default-jre -y
#sudo apt-get install default-jdk -y

#echo "👉 Done with installing Java 😎"

# To know java.home type: java -XshowSettings:properties -version
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre

# install maven
cd $HOME
wget https://downloads.apache.org/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz
tar xvfz apache-maven-3.6.3-bin.tar.gz
export M2_HOME=$HOME/apache-maven-3.6.3
export M2=$M2_HOME/bin
export PATH=$M2:$PATH

echo "👉 Done with installing Maven 😎"

# get Adam and install
cd $HOME
git clone https://github.com/bigdatagenomics/adam.git
cd adam
mvn install

echo "👉 Done with installing Adam 😎"

# get Cannoli and install
cd $HOME
git clone https://github.com/bigdatagenomics/cannoli.git
cd cannoli
mvn install

echo "👉 Done with installing Cannoli 😎"

# get Spark
cd $HOME
wget https://archive.apache.org/dist/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz
tar xvfz spark-2.4.5-bin-hadoop2.7.tgz
export SPARK_HOME=$HOME/spark-2.4.5-bin-hadoop2.7

echo "👉 Done with installing Spark 😎"

# Change permissions to /mydata
#sudo chown ${USER} /mydata
cd $DATA_DIR

#cp $PROJECT_NAME/hs38/* .
#echo "👉 Done copying the reference genome 😎"

#cp $PROJECT_NAME/fastq/*.fastq.gz .
#gunzip *.fastq.gz
#echo "👉 Done copying the .fastq files 😎"


#wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00128/sequence_read/SRR718072_1.filt.fastq.gz
#wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00128/sequence_read/SRR718072_2.filt.fastq.gz
#wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00128/sequence_read/SRR718072.filt.fastq.gz

echo "👉 Ready to invoke cannoli-shell 😎"
echo "./cannoli/bin/cannoli-shell  --executor-memory 50G --driver-memory 50G --conf spark.local.dir=/mydata"