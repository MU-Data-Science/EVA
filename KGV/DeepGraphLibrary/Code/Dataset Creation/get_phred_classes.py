import os, sys, glob
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# ap = argparse.ArgumentParser()
# ap.add_argument("-n", "--node", required=True, help="Node features")
# ap.add_argument("-e", "--edge", required=True, help="Edge features")

# args = vars(ap.parse_args())
# nodes_path = str(args['node']) if str(args['node']).endswith('.parquet') else f"{str(args['node'])}.parquet"
# edge_path = str(args['edge']) if str(args['edge']).endswith('.parquet') else f"{str(args['node'])}.parquet"

node_path = '/mydata/Updated/nodes'
edge_path = '/mydata/Updated/edges'

def concat_data(node_path):
    files = glob.glob(node_path)
    df = pd.concat([pd.read_parquet(f) for f in files], axis=0)
    return df

def get_centroids(node_path):
    df = concat_data(node_path)
    print(f"length of nodes_data: {len(df)}")

    kmeans = KMeans(n_clusters=7, random_state=0, max_iter=1000)
    df['cluster'] = kmeans.fit_predict(df[['phred_score']])
    centroids = kmeans.cluster_centers_
    print(centroids)

    print(df.groupby('cluster')['cluster'].count())

def get_cadd_class_mappings(phred_score):
    phred_score_map = {'0': 0.0, '1': 3.76, '2':  7.157, '3': 10.493, '4': 14.416, '5': 25.589}
    cadd_class = 0
    distance = [(k, abs(phred_score - v)) for k, v in phred_score_map.items()]
    distance.sort(key = lambda x:x[1])
    return distance[0][0]

new_cadd_classes = []
df = pd.DataFrame()
df_edges = pd.DataFrame()

def get_class(node_path, edge_path):
    print("Concatenating node data...")
    df = concat_data(node_path)
    df_edges = concat_data(edge_path)
    for _, row in df.iterrows():
        phred_score = row[11]
        cadd_class = get_cadd_class_mappings(phred_score)
        new_cadd_classes.append(cadd_class)

    df['cadd_class_using_phred'] = new_cadd_classes

    df.to_parquet('/mydata/combined_node_data_with_new_classes.parquet', index=False)
    df_edges.to_parquet('/mydata/combined_edge_data.parquet', index=False)

if __name__ == '__main__':
    # globals()[sys.argv[1]](sys.argv[2])
    globals()[sys.argv[1]]()

# get_class(node_path, edge_path)

