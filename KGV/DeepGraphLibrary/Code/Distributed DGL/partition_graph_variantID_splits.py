import pandas as pd
import glob
import torch
import dgl 
import os

node_path = '/mydata/dgl/general/input_data/nodes/'
edge_path = '/mydata/dgl/general/input_data/edges'

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

    print(f"Nodes length: {len(nodes_data)}")
    # Test set
    result = edges_data.loc[edges_data['dest']==39, ['variant_file', 'src']]
    result.rename(columns={'variant_file': 'variant', 'src': 'origin'}, inplace=True)
    test_set = pd.merge(result, nodes_data, on=['variant', 'origin'], how='inner')
    test_set.drop_duplicates(subset=['variant', 'origin', 'alt_genome', 'position', 'quality'], keep=False, inplace=True)
    
    print(f"Test set length: {len(test_set)}")

    # Train set: Validation set is untouched since it should be a combination of the other two sets
    train_set = pd.merge(nodes_data, test_set, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    print(f"Train set length: {len(train_set)}")

    test_size = len(test_set)
    train_size = len(train_set)

    new_nodes = pd.concat([train_set, test_set], axis=0)

    print(new_nodes.head(10))
    print(new_nodes.tail(10))

    node_features = torch.from_numpy(new_nodes.iloc[:, 2:-1].to_numpy(dtype='int64'))
    node_labels = torch.from_numpy(new_nodes['cadd_class'].astype('category').cat.codes.to_numpy()).to(torch.long)
    edge_features = torch.from_numpy(edges_data['predicate'].to_numpy(dtype='int64'))
    edges_src = torch.from_numpy(edges_data['src'].to_numpy(dtype='int64'))
    edges_dst = torch.from_numpy(edges_data['dest'].to_numpy(dtype='int64'))

    # node_features = nodes_data.iloc[:, 2:-1]
    # node_labels = nodes_data['cadd_class'].astype('category').cat.codes.to_numpy()
    # edge_features = edges_data['predicate']
    # edges_src = edges_data['src']
    # edges_dst = edges_data['dest']

    graph = dgl.graph((edges_src, edges_dst), num_nodes=new_nodes.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    n_nodes = new_nodes.shape[0]
    # n_nodes_train = train_set.shape[0]
    # n_nodes_test = test_set.shape[0]
    # n_train = int(n_nodes * 0.6)
    n_train = int(train_size * 0.9)
    # n_val = int(n_nodes * 0.2)
    n_val = int(train_size * 0.1) + int(test_size * 0.1)
    # print(f"Validation set length: {len(n_val)}")
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
