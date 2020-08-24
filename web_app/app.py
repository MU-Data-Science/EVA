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
    node_lst = cloudlab_setup.create_cluster(slice_name, nodes, site)

    # Setting up cluster config.
    cloudlab_setup.setup_cluster_config(node_lst)

    # Performing variant analysis.
    print("app.py :: execute :: Performing variant  analysis.")
    command = 'ssh -o "StrictHostKeyChecking no" %s@%s "\${HOME}/EVA/scripts/autorun_variant_analysis.sh hs38 %s %s %s"' % (constants.USER_NAME, node_lst[0], seq_1_url, seq_2_url, len(node_lst))
    out, err = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate(command.encode('utf-8'))
    print(out.decode('utf-8'))
    print "app.py :: execute :: Completed performing variant analysis."

    # Copying the vcf file to a file server(give a unique identifier.
    command = 'scp -o "StrictHostKeyChecking no" %s@%s:VA-${USER}-result-fbayes-output.vcf ${HOME}/apache-tomcat/webapps/download/%s-result-fbayes-output.vcf' % (constants.USER_NAME, node_lst[0], slice_name)
    out, err = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate(command.encode('utf-8'))
    print(out.decode('utf-8'))
    print "app.py :: execute :: Completed downloading the file."

    # Sending out the email with the download urls.
    command = 'mail -s "Variant Analysis Complete!" %s <<< "Hello,\n\nYour Variant Analysis job has been successfully completed. It can be downloaded from http://$(hostname -i):8080/download/%s \n\nThanks,\nEVA-Team"' % (email, slice_name + "-result-fbayes-output.vcf")
    out, err = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE).communicate(command.encode('utf-8'))
    print(out.decode('utf-8'))

    return "Successfully completed processing. Watch out for the email containing the links to download the .vcf."

@app.route("/")
def home():
  return render_template("index.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0')