'''
Developed by Shivika Prasanna on 02/16/2022.
Last updated on 03/22/2022.
Reads ntriples and annotates unique graph name. 
Working code. 
Run in terminal as: python3 Code/AnnotateGraphName.py -e <excel path> -n <annotated directory path>
excel_path = '/path/to/RNA-Sequence-Details.xlsx'
n3_path = '/path/to/VCF/annotated/N3'
'''

import os, csv
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--excel", required=True, help="excel file")
ap.add_argument("-n", "--ntriples", required=True, help="folder containing N3 files")

args = vars(ap.parse_args())
excel_path = str(args["excel"])
n3_path = str(args["ntriples"])

with open(excel_path, mode='rb') as xl:
    df = pd.read_excel(xl)

for n3_file in os.listdir(n3_path):
    if n3_file.endswith('.n3'):
        head = n3_file.split('.')[-4]
        print("head: ", head)
        if any(df['Run Accession ID'] == head) is True:
            accession_id = head
            graph_label = " <sg://" + accession_id + "> ."
            with open (os.path.join(n3_path, n3_file), 'r') as n3_file_in, open(os.path.join(n3_path, n3_file.replace('.n3', '.nq')), 'w') as n3_file_out:
                for i, line in enumerate(n3_file_in):
                    line = line.rstrip('. \n') + graph_label
                    print(line, file=n3_file_out)





        



