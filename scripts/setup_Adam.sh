#!/usr/bin/env bash

DATA_DIR=/mydata

# To know java.home type: java -XshowSettings:properties -version
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre

# install maven
cd $HOME
MAVEN_RELEASE="apache-maven-3.6.3-bin.tar.gz"
wget https://downloads.apache.org/maven/maven-3/3.6.3/binaries/${MAVEN_RELEASE}
tar xvfz ${MAVEN_RELEASE}
rm -rf ${MAVEN_RELEASE}
export M2_HOME=$HOME/apache-maven-3.6.3
export M2=$M2_HOME/bin
export PATH=$M2:$PATH

echo "👉 Done with installing Maven 😎"

# get Adam and install
cd $HOME
git clone https://github.com/bigdatagenomics/adam.git
cd adam
./scripts/move_to_scala_2.12.sh
mvn install

echo "👉 Done with installing Adam 😎"

# get Cannoli and install
cd $HOME
git clone https://github.com/bigdatagenomics/cannoli.git
cd cannoli
mvn install

echo "👉 Done with installing Cannoli 😎"

# get Spark
#cd $HOME
#wget https://archive.apache.org/dist/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz
#tar xvfz spark-2.4.5-bin-hadoop2.7.tgz
#export SPARK_HOME=$HOME/spark-2.4.5-bin-hadoop2.7

#echo "👉 Done with installing Spark 😎"

#cd $DATA_DIR

echo "👉 Ready to invoke cannoli-shell 😎"
echo "./cannoli/bin/cannoli-shell  --executor-memory 50G --driver-memory 50G --conf spark.local.dir=/mydata"