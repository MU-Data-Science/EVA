import os
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import json, pickle

import argparse

def concat_data(path):
    if os.path.isfile(path):
        df = pd.read_parquet(path)
    else:
        for root, dirs, files in os.walk(path):
            print(f"Inside first for {os.path.join(root, files[0])}")
            df = pd.read_parquet(os.path.join(root, files[0]))
            df.drop_duplicates(inplace=True)
            for f in files[1:]:
                print(f"Inside second for {os.path.join(root, f)}")
                df_temp = pd.read_parquet(os.path.join(root, f))
                df_temp.drop_duplicates(inplace=True)
                df = pd.concat([df, df_temp], axis=0)
    return df

def process_data(edge_path, pickle_file):

    dump = pickle.load(open(pickle_file, 'rb'))
    populated_df = concat_data(edge_path)

    new_populated_df = populated_df.loc[(populated_df['predicate'] == 'sg://0.99.11/vcf2rdf/variantId')]
    new_populated_df = new_populated_df.loc[new_populated_df['dest'] != 'None']

    print(len(new_populated_df))
    print(new_populated_df.head(25))

    origin_variantID_pairs = {}

    for idx, item in tqdm(new_populated_df.iterrows()):
        if item['dest'] is not None and item['dest'] != 'None':
            origin = item['src']
            variantID = item['dest']
            if variantID not in origin_variantID_pairs:
                origin_variantID_pairs[variantID] = set()
            origin_variantID_pairs[variantID].add(origin)

    pairs = {}
    for key, value in tqdm(origin_variantID_pairs.items()):
        origin_list = list(value)
        if len(origin_list) == 1:
            continue
        res = combinations(origin_list, 2)

        # with open(f"result.txt", 'a') as f:
        for pair in res:
            if pair not in pairs:
                pairs[pair] = 0
            pairs[pair] += 1

    mapped_pairs = {'src': [], 'predicate': [], 'dest': []}
    for pair, count in pairs.items():
        if pair[0] in dump['src_dest_dict'] and pair[1] in dump['src_dest_dict']:
            mapped_pairs['src'].append(dump['src_dest_dict'][pair[0]])
            mapped_pairs['predicate'].append(10)
            mapped_pairs['dest'].append(dump['src_dest_dict'][pair[1]])

    print("writing")
    pd.DataFrame(mapped_pairs).to_parquet('/mydata/dgl/general/new_edges_1.parquet', index=False)
    # pickle.dump(mapped_pairs, open("mapped_result.pkl", "wb"))
    return new_populated_df

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--edge", required=True, help="edge features")
    ap.add_argument("-p", "--picklefile", required=True, help="pickle file")

    args = vars(ap.parse_args())
    edge_path = str(args['edge'])
    pickle_file = str(args['picklefile'])

    print("Processing data!")
    process_data(edge_path, pickle_file)