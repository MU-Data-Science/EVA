import random
import string
import cloudlab_setup
import constants
import subprocess

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/execute')
def execute_cluster():
    # Reading data from the request.
    nodes = int(request.args.get("nodes"))
    site = request.args.get("site")
    email = request.args.get("email")
    seq_1_url = request.args.get("seq_1_url")
    seq_2_url = request.args.get("seq_2_url")
    ref = request.args.get("ref")

    # Slice name
    slice_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    print("app.py :: execute_cluster :: Slice name created :: ", slice_name)

    # Setting up the cluster.
    node_lst = cloudlab_setup.create_cluster(slice_name, nodes, site)

    # Setting up cluster config.
    cloudlab_setup.setup_cluster_config(node_lst)

    # Performing variant analysis.
    execute_va(node_lst[0], seq_1_url, seq_2_url, ref, len(node_lst), slice_name)

    # Sending out the email with the download urls.
    command = 'mail -s "Variant Analysis Complete!" %s <<< "Hello,\n\nYour Variant Analysis job has been successfully completed. It can be downloaded from http://$(hostname -i):8080/download/%s \n\nThanks,\nEVA-Team"' % (email, slice_name + "-result-fbayes-output.vcf")
    out, err = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate(command.encode('utf-8'))
    print(out.decode('utf-8'))

    return "Successfully completed processing. Watch out for the email containing the links to download the .vcf."


def execute_va(node, seq_1_url, seq_2_url, ref, cluster_size, id):
    # Performing variant analysis.
    print("app.py :: execute :: Performing variant  analysis.")
    command = 'ssh -o "StrictHostKeyChecking no" %s@%s "\${HOME}/EVA/scripts/autorun_variant_analysis.sh %s %s %s %s"' % (constants.USER_NAME, node, ref, seq_1_url, seq_2_url, cluster_size)
    out, err = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate(command.encode('utf-8'))
    print(out.decode('utf-8'))
    print "app.py :: execute_va :: Completed performing variant analysis."

    # Copying the vcf file to a file server(give a unique identifier.
    command = 'scp -o "StrictHostKeyChecking no" %s@%s:VA-${USER}-result-fbayes-output.vcf ${HOME}/apache-tomcat/webapps/download/%s-result-fbayes-output.vcf' % (constants.USER_NAME, node, id)
    out, err = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate(
        command.encode('utf-8'))
    print(out.decode('utf-8'))
    print "app.py :: execute_va :: Completed downloading the file."


@app.route('/execute')
def execute():
    # Reading data from the request.
    email = request.args.get("email")
    seq_1_url = request.args.get("seq_1_url")
    seq_2_url = request.args.get("seq_2_url")
    ref = request.args.get("ref")

    # Creating an id for the experiment.
    id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    print("app.py :: execute :: Experiment Id :: ", id)

    # Performing variant analysis.
    execute_va(constants.MASTER_NODE, seq_1_url, seq_2_url, ref, constants.CLUSTER_SIZE, id)

    # Sending out the email with the download urls.
    command = 'mail -s "Variant Analysis Complete!" %s <<< "Hello,\n\nYour Variant Analysis job has been successfully completed. It can be downloaded from http://$(hostname -i):8080/download/%s \n\nThanks,\nEVA-Team"' % (email, id + "-result-fbayes-output.vcf")
    out, err = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate(command.encode('utf-8'))
    print(out.decode('utf-8'))

    return "Successfully completed processing. Watch out for the email containing the links to download the .vcf."


@app.route("/standalone")
def standalone():
  return render_template("standalone.html")


@app.route("/cluster")
def cluster():
  return render_template("cluster.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0')