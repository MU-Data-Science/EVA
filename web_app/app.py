import random
import string
import cloudlab_setup
import constants
import subprocess

from flask import Flask, request, render_template

app = Flask(__name__)


def execute_va(node, seq_1_url, seq_2_url, ref, cluster_size, exp_id):
    # Performing variant analysis.
    print("app.py :: execute :: Performing variant  analysis.")
    command = 'ssh -o "StrictHostKeyChecking no" %s@%s "\${HOME}/EVA/scripts/autorun_variant_analysis.sh %s %s %s %s"' % (
    constants.USER_NAME, node, ref, seq_1_url, seq_2_url, cluster_size)
    out, err = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate(command.encode('utf-8'))
    print(out.decode('utf-8'))
    print "app.py :: execute_va :: Completed performing variant analysis."

    # Copying the vcf file to a file server(give a unique identifier.
    command = 'scp -o "StrictHostKeyChecking no" %s@%s:VA-${USER}-result-fbayes-output.vcf.zip ${HOME}/apache-tomcat/webapps/download/%s-result-fbayes-output.vcf.zip' % (
    constants.USER_NAME, node, exp_id)
    out, err = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate(
        command.encode('utf-8'))
    print(out.decode('utf-8'))
    print "app.py :: execute_va :: Completed downloading the file."


def get_uptime():
    command = "uptime"
    out, err = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate(command.encode('utf-8'))
    print("get_uptime :: ", out.decode('utf-8').strip().split(" ")[-1])
    return out.decode('utf-8').strip().split(" ")[-1]


@app.route('/execute_standalone')
def execute_standalone():
    # Reading data from the request.
    nodes = constants.DEFAULT_CLUSTER_SIZE
    if request.args.get("nodes") is not None:
        nodes = int(request.args.get("nodes"))
    site = request.args.get("site")
    email = request.args.get("email")
    seq_1_url = request.args.get("seq_1_url")
    seq_2_url = request.args.get("seq_2_url")
    ref = request.args.get("ref")

    # Slice name
    slice_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    print("app.py :: execute_standalone :: Slice name created :: ", slice_name)

    # Setting up the cluster.
    node_lst = cloudlab_setup.create_cluster(slice_name, nodes, site)

    # Setting up cluster config.
    cloudlab_setup.setup_cluster_config(node_lst)

    # Performing variant analysis.
    execute_va(node_lst[0], seq_1_url, seq_2_url, ref, len(node_lst), slice_name)

    # Sending out the email with the download urls.
    command = 'mail -s "Variant Analysis Complete!" %s <<< "Hello,\n\nYour Variant Analysis job has been successfully completed. It can be downloaded from http://$(hostname -i):8080/download/%s \n\nThanks,\nEVA-Team"' % (email, slice_name + "-result-fbayes-output.vcf.zip")
    out, err = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate(command.encode('utf-8'))
    print(out.decode('utf-8'))

    return "Successfully completed processing. Watch out for the email containing the links to download the .vcf."


@app.route('/execute_cluster')
def execute_cluster():
    # Reading data from the request.
    email = request.args.get("email")
    seq_1_url = request.args.get("seq_1_url")
    seq_2_url = request.args.get("seq_2_url")
    ref = request.args.get("ref")

    # Creating an id for the experiment.
    exp_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    print("app.py :: execute_cluster :: Experiment Id :: ", exp_id)

    # Obtaining the cluster uptime.
    uptime = float(get_uptime())
    if uptime > constants.UPTIME:
        # Perform variant analysis by creating a separate cluster.
        execute_standalone()
    else:
        # Performing variant analysis on the same cluster.
        execute_va(constants.MASTER_NODE, seq_1_url, seq_2_url, ref, constants.DEFAULT_CLUSTER_SIZE, exp_id)

        # Sending out the email with the download urls.
        command = 'mail -s "Variant Analysis Complete!" %s <<< "Hello,\n\nYour Variant Analysis job has been successfully completed. It can be downloaded from http://$(hostname -i):8000/download/%s \n\nThanks,\nEVA-Team"' % (email, exp_id + "-result-fbayes-output.vcf")
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