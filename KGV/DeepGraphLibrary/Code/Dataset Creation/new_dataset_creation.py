'''
Latest version of the working code.
java -server -Xmx8g -jar /mydata/dgl/blazegraph.jar
Run run.sh for BulkDataLoader
'''

import os
import json
import pandas as pd
from pymantic import sparql
import requests
from collections import OrderedDict
from tqdm import tqdm

import gc
import subprocess

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-ip", "--ip", required=True, help="IP address connecting us to Blazegraph")
ap.add_argument("-e", "--edge", required=True, help="To write mapped edge features")
ap.add_argument("-n", "--node", required=True, help="To write mapped node features")

args = vars(ap.parse_args())

blazegraph_ip_address = str(args['ip'])
edges_path = str(args['edge']) if str(args['edge']).endswith('.parquet') else f"{str(args['edge'])}.parquet"
nodes_path = str(args['node']) if str(args['node']).endswith('.parquet') else f"{str(args['node'])}.parquet"

url = f'http://{blazegraph_ip_address}:9999/blazegraph/sparql'

headers = {
    'Accept': 'application/json',
}

def get_edges_data(edges_path):
    raw_edge_lst = []
    variant_origin_tuples_list = set()

    print("👉 Connected to BlazeGraph, fetching edges now!")

    data_1 = {
        'query': 'SELECT DISTINCT ?accession_id ?origin ?variant_id_uri ?variant_id WHERE { VALUES ?variant_id_uri { <sg://0.99.11/vcf2rdf/variantId> } GRAPH ?accession_id { ?origin ?variant_id_uri ?variant_id . } OPTIONAL { ?origin ?variant_id_uri ?variant_id . } }',
    }
    result_1 = requests.post(url, headers=headers, data=data_1)
    print(f"👉 Finished fetching {len(result_1.json()['results']['bindings'])} edges!")

    for key in result_1.json()['results']['bindings']:
        accession_id = key['accession_id']['value'].split('//')[-1]
        origin = key['origin']['value']
        variant_id_uri =  key['variant_id_uri']['value']
        if 'variant_id' not in key:
            key['variant_id'] = {'type': 'uri', 'value': 'None'}

        variant_id = key['variant_id']['value']
        
        src_dst_list = ['accession_id', 'origin', 'variant_id']
        edge_list = ['variant_id_uri']

        raw_edge_lst.append((accession_id, origin, variant_id_uri, variant_id))

    dump_edges_data(raw_edge_lst, edges_path.replace('.parquet', '_raw.parquet'))
    print("Completed fetching edge features!")
    return raw_edge_lst

def dump_edges_data(id_lst, edges_path):
    print("👉 Writing mapped edges to CSV.")
    edge_cols = ['variant_file', 'src', 'predicate', 'dest']
    df_mapped_edges = pd.DataFrame(id_lst, columns=edge_cols) 
    df_mapped_edges.to_parquet(edges_path, index=False)

def get_nodes_data(nodes_path):
    data_2 = {
        'query': 'SELECT DISTINCT ?ref_genome WHERE { ?variant <http://sg.org/has_ref_genome> ?ref_genome . }',
    }
    result_2 = requests.post(url, headers=headers, data=data_2)
    print("👉 Finished fetching node features!")

    raw_node_lst = []
    data_5_set = set()

    for item in result_2.json()['results']['bindings']:
        ref_genome = item['ref_genome']['value']
        print(f"Fetching results for {ref_genome}")
        data_3 = {
            'query': f"SELECT DISTINCT ?alt_genome WHERE {{ ?variant <http://sg.org/has_alt_genome> ?alt_genome . ?variant <http://sg.org/has_ref_genome> '{ref_genome}' . }}",
        }
        result_3 = requests.post(url, headers=headers, data=data_3)
        print(f"👉Found {len(result_3.json()['results']['bindings'])}")
        
        for i in result_3.json()['results']['bindings']:
            alt_genome = i['alt_genome']['value']
            print(f"Fetching results for ref_genome {ref_genome} and alt_genome {alt_genome}")

            data_4 = {
                'query': f"SELECT DISTINCT ?pos WHERE {{ ?variant <http://sg.org/has_alt_genome> '{alt_genome}' . ?variant <http://sg.org/has_ref_genome> '{ref_genome}' . ?variant <http://sg.org/has_pos> ?pos . }}",
            }
            result_4 = requests.post(url, headers=headers, data=data_4)
            
            for i_result_4 in result_4.json()['results']['bindings']:
                pos = i_result_4['pos']['value']
                origin_set = []
                result_5 = {}
                data_5 = {
                    'query': f"SELECT DISTINCT ?variant ?origin ?quality ?chromosome ?ann ?phred_score ?raw_scores WHERE {{ ?variant <http://sg.org/has_alt_genome> '{alt_genome}' . ?origin <sg://0.99.11/vcf2rdf/variant/ALT> <sg://0.99.11/vcf2rdf/sequence/{alt_genome}> . ?variant <http://sg.org/has_ref_genome> '{ref_genome}' . ?origin <sg://0.99.11/vcf2rdf/variant/REF> <sg://0.99.11/vcf2rdf/sequence/{ref_genome}> . ?variant <http://sg.org/has_pos> {pos} . ?origin <http://biohackathon.org/resource/faldo#position> {pos} . ?origin <sg://0.99.11/vcf2rdf/variant/QUAL> ?quality . ?origin <http://biohackathon.org/resource/faldo#reference> ?chromosome . ?origin <sg://0.99.11/vcf2rdf/info/ANN> ?ann . ?variant <http://sg.org/has_cadd_scores> ?cadd_scores . ?cadd_scores <http://sg.org/has_raw_score> ?raw_scores . ?cadd_scores <http://sg.org/has_phred> ?phred_score . }}", 
                } 
                result_5 = requests.post(url, headers=headers, data=data_5)
                for key in result_5.json()['results']['bindings']:
                    accession_id = key['variant']['value'].split('/')[-3]
                    origin = key['origin']['value']
                    origin_set.append(f"<{origin}>")
                    quality = key['quality']['value']
                    chromosome = key['chromosome']['value'].split('/')[-1]
                    ann_allele = key['ann']['value'].split('|')[0]
                    ann_annotation = key['ann']['value'].split('|')[1]
                    ann_impact = key['ann']['value'].split('|')[2]
                    ann_gene_name = key['ann']['value'].split('|')[3]
                    ann_gene_id = key['ann']['value'].split('|')[4]
                    phred_score = float(key['phred_score']['value'])
                    raw_scores = float(key['raw_scores']['value'])

                    data_5_item = (pos, alt_genome, ref_genome, chromosome, origin, accession_id, quality, phred_score) 
                    if data_5_item not in data_5_set:
                        data_5_set.add(data_5_item)
                    else:
                        print("Duplicate found: ", key)
                
                    raw_node_lst.append([accession_id, origin, alt_genome, ref_genome, pos, quality, str(chromosome), ann_allele, ann_annotation, ann_impact, ann_gene_name, ann_gene_id, phred_score, raw_scores])
                    
                    if len(raw_node_lst) % 1000 == 0:
                        print("Wrote: ", len(raw_node_lst))

    print("Completed fetching node features!")
    
    dump_node_data(raw_node_lst, nodes_path.replace('.parquet', '_raw.parquet'))

    return raw_node_lst

def dump_node_data(id_node_lst, nodes_path):
    print("👉 Dumping mapped nodes to CSV.")
    node_cols=['accession_id', 'origin', 'alt_genome', 'ref_genome', 'position', 'quality', 'chromosome', 'ann_allele', 'ann_annotation', 'ann_impact', 'ann_gene_name', 'ann_gene_id', 'phred_score', 'raw_scores']
    df_1 = pd.DataFrame(id_node_lst, columns=node_cols)
    df_1.to_parquet(nodes_path, index=False)
    print("Completed dumping mapped nodes to CSV!")

raw_edge_lst = get_edges_data(edges_path)
raw_node_lst = get_nodes_data(nodes_path)
# subprocess.run(['/mydata/send_email.sh'], input='Done', text=True)