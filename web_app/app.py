import random
import string
import cloudlab_setup
import constants
import subprocess

from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/execute')
def execute():
    # Reading data from the request.
    nodes = int(request.args.get("nodes"))
    site = request.args.get("site")
    email = request.args.get("email")
    seq_1_url = request.args.get("seq_1_url")
    seq_2_url = request.args.get("seq_2_url")

    # Slice name
    slice_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    print("app.py :: execute :: Slice name created :: ", slice_name)

    # Setting up the cluster.
    # node_lst = cloudlab_setup.create_cluster(slice_name, nodes, site)

    # Setting up cluster config.
    node_lst = ['c220g1-030822.wisc.cloudlab.us', 'c220g1-030621.wisc.cloudlab.us']
    cloudlab_setup.setup_cluster_config(node_lst)

    # Performing variant analysis.
    print("app.py :: execute :: Performing variant  analysis.", slice_name)
    command = 'ssh -o "StrictHostKeyChecking no" %s@%s "${HOME}/EVA/scripts/autorun_variant_analysis.sh hs38 %s %s %s"' % (constants.USER_NAME, node_lst[0], seq_1_url, seq_2_url, len(node_lst))
    process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = process.communicate(command.encode('utf-8'))
    print(out.decode('utf-8'))
    print "cloudlab_setup.py :: setup_cluster_config :: Completed setting up Hadoop and Spark on the nodes."

    # Copy the vcf file to a file server(give a unique identifier.

    return "Successfully completed processing. Watch out for the email containing the links to download the .vcf."

@app.route("/")
def home():
  return render_template("index.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0')