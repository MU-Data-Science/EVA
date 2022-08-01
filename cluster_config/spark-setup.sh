#!/bin/bash
MASTER_NODE_ID="$1"
MSTR='vm0'
data_dir="$2"
share_dir="$3"
spark_ver="$4"
hadoop_ver="$5"

NUMREP=1

# check if openssl is installed
ISOSSL=`which openssl`
[ -z "$ISOSSL" ] && sudo apt-get --yes install openssl 

# generate password
secret_file="spark-secret.txt"
openssl rand -base64 100 | tr -dc 'a-zA-Z0-9' | fold -w 128 | head -n 1 > $secret_file
SPARK_SECRET=$(head -1 $secret_file)

cluster_prefix="$share_dir/"

spark_prefix="$cluster_prefix/spark_$MASTER_NODE_ID"
MSTR="vm0"

rm -Rf "$spark_prefix"
mkdir -p "$spark_prefix"

spark_file_name="spark-"$spark_ver"-bin-hadoop"$hadoop_ver".tgz"

# If Spark 2.4.* is used w/ scala 2.12
if [[ $hadoop_ver == "0.0" ]]; then
    spark_file_name="spark-"$spark_ver"-bin-without-hadoop-scala-2.12.tgz"
fi

if [ ! -f $share_dir/EVA_Tools/$spark_file_name ]; then
    wget https://archive.apache.org/dist/spark/spark-$spark_ver/$spark_file_name -P $share_dir/EVA_Tools/
fi
tar zxf $share_dir/EVA_Tools/$spark_file_name -C "$spark_prefix" --strip-components 1

SPARK_DEFAULTS_FILE="$spark_prefix/conf/spark-defaults.conf"
echo "
#spark.master                spark://$MSTR:7077
#spark.driver.memory          20g
spark.executor.memory        12g
spark.executor.cores         4
spark.shuffle.blockTransferService nio
spark.network.timeout        840000
spark.shuffle.io.maxRetries  10
spark.shuffle.io.retryWait   30
spark.eventLog.dir           hdfs://$MSTR:9000/spark-events
spark.eventLog.enabled       true
spark.authenticate           true
spark.authenticate.secret    $SPARK_SECRET
spark.ui.enabled             false
" > $SPARK_DEFAULTS_FILE

SPARK_ENV_FILE="$spark_prefix/conf/spark-env.sh"
echo "
export JAVA_HOME=/usr
export SPARK_MASTER_HOST=$MSTR
export SPARK_PUBLIC_DNS=$MSTR
export SPARK_LOCAL_DIRS=$data_dir/spark-tmp
export SPARK_WORKER_OPTS=\"-Dspark.worker.cleanup.enabled=true -Dspark.worker.cleanup.interval=7200 -Dspark.worker.cleanup.appDataTtl=1800\"
" > $SPARK_ENV_FILE

# If Spark 2.4.* is used w/ scala 2.12 and no Hadoop
if [[ $hadoop_ver == "0.0" ]]; then
    echo "export SPARK_DIST_CLASSPATH=$("$data_dir"/hadoop/bin/hadoop classpath)" >> $SPARK_ENV_FILE
fi

cp "$cluster_prefix/hadoop_$MASTER_NODE_ID/etc/hadoop/slaves" "$spark_prefix/conf/slaves"

rm "$secret_file"
