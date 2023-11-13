# import heartrate 
# heartrate.trace(browser=True)

import os

import pandas as pd
import pickle

from tqdm import tqdm
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-e", "--edge", required=True, help="Raw edges file")
ap.add_argument("-n", "--node", required=True, help="Raw nodes file")
ap.add_argument("-d", "--dump", required=True, help="pickle to write the edges and node features mappings")

args = vars(ap.parse_args())
raw_edges_path = str(args['edge'])
raw_nodes_path = str(args['node'])
dump_path = str(args['dump'])

def setup(dump_path):
    src_dest_dict = {}
    edge_dict = {}
    genome_dict = {'ref_genome': {}, 'alt_genome': {}, 'position': {}, 'quality': {}, 'chromosome': {}, 'ann_allele': {}, 'ann_annotation': {}, 'ann_impact': {}, 'ann_gene_name': {}, 'ann_gene_id': {}}

    if os.path.exists(dump_path):
        pickle_map = pickle.load(open(dump_path, 'rb'))
        genome_dict = pickle_map['genome_dict']
        src_dest_dict = pickle_map['src_dest_dict']
        edge_dict = pickle_map['edge_dict']
        print("Finished initial setup!")
    else:
        print("Pickle file does not exist, creating new file!")

    return src_dest_dict, edge_dict, genome_dict

def update_genome_dict(key_value_dict, genome_dict, genome_dict_counter):
    # key_value_dict = {'ref_genome': ref_genome, 'alt_genome': alt_genome, 'position': position, 'quality': key['quality']['value'], 'chromosome': chromosome, 'ann_allele': ann_allele, 'ann_annotation': ann_annotation, 'ann_impact': ann_impact, 'ann_gene_name': ann_gene_name, 'ann_gene_id': ann_gene_id}
    for key, value in key_value_dict.items():
        if value not in genome_dict[key]:
            genome_dict[key][value] = genome_dict_counter
            genome_dict_counter += 1
    return genome_dict_counter

def update_dicts(key_value_dict, update_dict):
    for _, value in key_value_dict.items():
        if value not in update_dict:
            update_dict[value] = len(update_dict)

def map_edges_data(raw_edges_path, src_dest_dict, edge_dict):
    df_edge = pd.read_parquet(raw_edges_path)
    id_lst = []
    lst = []
    variant_origin_tuples_list = set()

    for _, row in df_edge.iterrows():

        accession_id = row['variant_file']
        src = row['src']
        dest = row['dest']
        edge = row['predicate']

        src_dest_value_dict = {'src': src, 'dest': dest}
        edge_value_dict = {'predicate': edge}

        update_dicts(src_dest_value_dict, src_dest_dict)
        update_dicts(edge_value_dict, edge_dict)

        id_lst.append((accession_id, src_dest_dict[src], edge_dict[edge], src_dest_dict[dest]))
        variant_origin_tuples_list.add((accession_id, src_dest_dict[dest]))

    dump_edges_data(id_lst, raw_edges_path.replace('_raw.parquet', '.parquet'))
    print("Completed fetching edge features!")
    return src_dest_dict, edge_dict, variant_origin_tuples_list

def dump_edges_data(id_lst, edges_path):
    print("👉 Writing mapped edges to CSV.")
    edge_cols = ['accession_id', 'src', 'predicate', 'dest']
    df_mapped_edges = pd.DataFrame(id_lst, columns=edge_cols) 
    df_mapped_edges.to_parquet(edges_path, index=False)

def get_genome_dict_max(genome_dict):
    p = []
    for k, v in genome_dict.items():
        if len(v) > 0:
            p.append(max(v.values()))
    return max(p) if p != [] else 0

def get_nodes_data(raw_nodes_path, src_dest_dict, genome_dict):
    df_node = pd.read_parquet(raw_nodes_path)
    id_node_lst = []
    DEFAULT_ORIGIN_VALUE = 0
    chrom_dict = {'X': 23, 'Y': 24, 'MT': 25}

    genome_dict_counter = get_genome_dict_max(genome_dict)

    print(genome_dict_counter)

    for _, row in tqdm(df_node.iterrows()):
        accession_id = row['accession_id']
        origin = row['origin']
        alt_genome = row['alt_genome']
        ref_genome = row['ref_genome']
        position = row['position']
        quality = row['quality']
        chromosome = row['chromosome']
        ann_allele = row['ann_allele']
        ann_annotation = row['ann_annotation']
        ann_impact = row['ann_impact']
        ann_gene_name = row['ann_gene_name']
        ann_gene_id = row['ann_gene_id']
        phred_score = row['phred_score']
        raw_scores = row['raw_scores']
            
        key_origin = src_dest_dict.get(origin, DEFAULT_ORIGIN_VALUE)          

        key_value_dict = {'alt_genome': alt_genome, 'ref_genome': ref_genome, 'position': position, 'quality': quality, 'chromosome': chromosome, 'ann_allele': ann_allele, 'ann_annotation': ann_annotation, 'ann_impact': ann_impact, 'ann_gene_name': ann_gene_name, 'ann_gene_id': ann_gene_id}

        genome_dict_counter = update_genome_dict(key_value_dict, genome_dict, genome_dict_counter)
                    
        id_node_lst.append([accession_id, key_origin, genome_dict['alt_genome'][alt_genome], genome_dict['ref_genome'][ref_genome], genome_dict['position'][position], genome_dict['quality'][quality], genome_dict['chromosome'][chromosome], genome_dict['ann_allele'][ann_allele], genome_dict['ann_annotation'][ann_annotation], genome_dict['ann_impact'][ann_impact], genome_dict['ann_gene_name'][ann_gene_name], genome_dict['ann_gene_id'][ann_gene_id], round(phred_score), round(raw_scores), phred_score, raw_scores])
                    
    print("Completed fetching node features!")
    node_cols=['accession_id', 'origin', 'alt_genome', 'ref_genome', 'position', 'quality', 'chromosome', 'ann_allele', 'ann_annotation', 'ann_impact', 'ann_gene_name', 'ann_gene_id', 'rounded_phred_score', 'rounded_raw_score', 'actual_phred_score', 'actual_raw_scores']

    df_1 = pd.DataFrame(id_node_lst, columns=node_cols)
    df_1.to_parquet(raw_nodes_path.replace('_raw.parquet', '.parquet'), index=False)

    return id_node_lst, genome_dict

def merge_and_dump_node_data(id_node_lst, variant_origin_tuples_list, nodes_path):
    print("👉 Dumping mapped nodes to CSV.")
    node_cols=['accession_id', 'origin', 'alt_genome', 'ref_genome', 'position', 'quality', 'chromosome', 'ann_allele', 'ann_annotation', 'ann_impact', 'ann_gene_name', 'ann_gene_id', 'rounded_phred_score', 'rounded_raw_score','phred_score', 'raw_scores']
    df_1 = pd.DataFrame(id_node_lst, columns=node_cols)
    df_2 = pd.DataFrame([[item[0], item[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for item in variant_origin_tuples_list], columns=node_cols)
    result_df = pd.concat([df_1, df_2], axis=0)
    result_df.to_parquet(nodes_path, index=False)
    print("Completed dumping mapped nodes to CSV!")

def dump_ids(dump_path, genome_dict, src_dest_dict, edge_dict):
    print("👉 Dumping mapped features to pickle.")
    mapped = {'genome_dict': genome_dict, 'src_dest_dict': src_dest_dict, 'edge_dict': edge_dict}
    pickle.dump(mapped, open(dump_path, 'wb'))
    print("Completed dumping mapped features to pickle!")


print("---> Setting up dump pickle file.. ")
src_dest_dict, edge_dict, genome_dict = setup(dump_path)
print("---> Mapping edges data.. ")
src_dest_dict, edge_dict, variant_origin_tuples_list = map_edges_data(raw_edges_path, src_dest_dict, edge_dict)
print("---> Mapping nodes data.. ")
id_node_lst, genome_dict = get_nodes_data(raw_nodes_path, src_dest_dict, genome_dict)
print("---> Dumping all mapped data into a pickle file.. ")
dump_ids(dump_path, genome_dict, src_dest_dict, edge_dict)
print("---> Finally merging nodes from edges data with nodes data.. ")
merge_and_dump_node_data(id_node_lst, variant_origin_tuples_list, raw_nodes_path.replace('_raw.parquet', '.parquet'))
print("---> Process completed!")