import pandas as pd
import glob
import torch
import dgl 
import os

node_path = '/mydata/dgl/general/small_genomics_data/nodes'
edge_path = '/mydata/dgl/general/small_genomics_data/edges'

def downsize_data(df):
    INT32_LIMIT = 2147483647
    for col in df.columns:
        if 'int' in str(df[col].dtypes):
            if df[col].max() <= INT32_LIMIT:
                df[col] = df[col].astype('int32')
    return df

def concat_data(filepath):
    for root, dirs, files in os.walk(filepath):
        print(f"Inside first for {os.path.join(root, files[0])}")
        df = pd.read_parquet(os.path.join(root, files[0]))
        df = downsize_data(df)
        for f in files[1:]:
            print(f"Inside second for {os.path.join(root, f)}")
            df_temp = pd.read_parquet(os.path.join(root, f))
            df_temp = downsize_data(df_temp)
            df = pd.concat([df, df_temp], axis=0)
    return df
    # df = pd.concat([pd.read_parquet(f) for f in files], axis=0)

def load_genomics_graph():
    nodes_data = concat_data(node_path)
    edges_data = concat_data(edge_path)

    node_features = torch.from_numpy(nodes_data.iloc[:, 2:-1].to_numpy(dtype='int64'))
    node_labels = torch.from_numpy(nodes_data['cadd_class'].astype('category').cat.codes.to_numpy()).to(torch.long)
    edge_features = torch.from_numpy(edges_data['predicate'].to_numpy(dtype='int64'))
    edges_src = torch.from_numpy(edges_data['src'].to_numpy(dtype='int64'))
    edges_dst = torch.from_numpy(edges_data['dest'].to_numpy(dtype='int64'))

    # node_features = nodes_data.iloc[:, 2:-1]
    # node_labels = nodes_data['cadd_class'].astype('category').cat.codes.to_numpy()
    # edge_features = edges_data['predicate']
    # edges_src = edges_data['src']
    # edges_dst = edges_data['dest']

    graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    n_nodes = nodes_data.shape[0]
    n_train = int(n_nodes * 0.6)
    n_val = int(n_nodes * 0.2)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    return graph
