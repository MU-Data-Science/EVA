# Imports.
import os
import subprocess
import time

# Constants.
USER = "arung"
HOME = "/users/"+USER
GENOME_DIR = "/nfs/GenomeData"
DATA_DIR = "/mydata"
EVA_SCRIPTS_DIR = HOME+"/EVA/scripts"
CLUSTER_SIZE = "16"

# Setting a counter.
counter = 0;

# Iterating over all the files in the given directory.
for dir in os.listdir(GENOME_DIR):
    # Identifying empty directories.
    if len(dir) == 0:
        print("collect_stats.py :: Empty directory encountered :: ", dir)
    else:
        # Iterating over the sequences.
        for file in os.listdir(GENOME_DIR + "/" + dir):
            if file.endswith("_1.filt.fastq.gz"):
                try:
                    # Obtaining the sequence id.
                    seq_id = file.split("_1.filt.fastq.gz")[0]
                    print("collect_stats.py :: Processing sequence :: ", seq_id, ":: in ::", (GENOME_DIR + "/" + dir))

                    # Updating the counter
                    counter = counter + 1

                    # Uploading the file to HDFS.
                    print("collect_stats.py :: Uploading the sequence files to HDFS for :: ", seq_id)
                    upload_1 = "hdfs dfs -copyFromLocal %s /" % (GENOME_DIR + "/" + dir + "/" + seq_id + "_1.filt.fastq.gz")
                    upload_2 = "hdfs dfs -copyFromLocal %s /" % (GENOME_DIR + "/" + dir + "/" + seq_id + "_2.filt.fastq.gz")

                    process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                    out, err = process.communicate(upload_1.encode('utf-8'))
                    print(out.decode('utf-8'))

                    process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                    out, err = process.communicate(upload_2.encode('utf-8'))
                    print(out.decode('utf-8'))

                    # Noting the start time.
                    start_time = time.time()

                    # Executing variant analysis.
                    print("collect_stats.py :: Performing variant analysis for sequence :: ", seq_id)
                    va_cmd = "%s/run_variant_analysis_adam.sh hs38 hdfs://vm0:9000/%s_1.filt.fastq.gz hdfs://vm0:9000/%s_2.filt.fastq.gz %s" % (EVA_SCRIPTS_DIR, seq_id, seq_id, CLUSTER_SIZE)
                    process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                    out, err = process.communicate(va_cmd.encode('utf-8'))
                    print(out.decode('utf-8'))

                    # Printing the execution time.
                    print("collect_stats.py :: Time taken to preocess the sequence :: %s :: was %s seconds." % (seq_id, time.time() - start_time))

                    # Renaming the file.
                    rename_cmd = "mv %s/VA-%s-result-gatk-spark-output.vcf %s/%s-result-gatk-spark-output.vcf" % (HOME, USER, DATA_DIR, seq_id)
                    process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                    out, err = process.communicate(rename_cmd.encode('utf-8'))
                    print(out.decode('utf-8'))

                    # Removing the file from HDFS.
                    print("collect_stats.py :: Deleting the sequence files from HDFS for :: ", seq_id)
                    del_1 = "hdfs dfs -rm /%s" % (seq_id + "_1.filt.fastq.gz")
                    del_2 = "hdfs dfs -rm /%s" % (seq_id + "_2.filt.fastq.gz")

                    process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                    out, err = process.communicate(del_1.encode('utf-8'))
                    print(out.decode('utf-8'))

                    process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
                    out, err = process.communicate(del_2.encode('utf-8'))
                    print(out.decode('utf-8'))

                    print("collect_stats.py :: Completed processing for sequence :: ", seq_id)
                except:
                    print("collect_stats.py :: Exception encountered while processing sequence :: ", seq_id)

                if counter == 25:
                    print("collect_stats.py :: Completed processing 25 sequences.")
                    exit(1)
