'''
Developed by Shivika Prasanna on 08/22/2022.
Last updated on 08/22/2022.
Extract Gene ID from ANN and create NQ triples.
Working code. 
Run in terminal as: $ python3 GeneIDasTriple.py -i <input directory> -o <output file>
'''

import os
import argparse
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="input path to directory")
ap.add_argument("-o", "--output", required=True, help="output file")

args = vars(ap.parse_args())
input_directory = str(args['input'])
if str(args['output']).endswith('nq'):
    gene_id_file_path = str(args['output'])
else:
    print("Output file must have .nq extension!")
    break

for root, dirs, files in os.walk(input_directory):
    for filename in files:
        filepath = os.path.join(root, filename)
        if filepath.endswith('.nq'):
            print("Processing file {} ".format(filepath))

            selected_lines = [line for line in open(filepath) if "<sg://0.99.11/vcf2rdf/info/ANN>" in line and not line.startswith('<sg://0.99.11/vcf2rdf/info/ANN>')]

            for line in selected_lines:
                origin = line.split(' ')[0]
                ann = line.split(' ')[1]
                ann_values = line.split(' ')[2]
                named_graph = line.split(' ')[3]
                # gene_uri = ann.replace('>','/GENE>')
                gene_name = ann_values.split('|')[3]
                gene_name_uri = ann.replace('>','/GENE_NAME>')
                gene_id = ann_values.split('|')[4]
                gene_id_uri = ann.replace('>','/GENE_ID>')
                # print("<{} \n {} \n {} \n {}>".format(origin, ann, gene_id, named_graph))
                # if gene_id is not "":
                #     print(ann, gene_id_uri, gene_id, named_graph)

                with open(gene_id_file_path, 'a') as gene_id_file:
                    if gene_id is not "":
                        gene_id_file.write(f"{origin} {gene_id_uri} \"{gene_id}\"^^<http://www.w3.org/2001/XMLSchema#string> {named_graph} .\n")
                        gene_id_file.write(f"{origin} {gene_name_uri} \"{gene_name}\"^^<http://www.w3.org/2001/XMLSchema#string> {named_graph} .\n")