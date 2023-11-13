import pandas as pd
import glob
import torch
import dgl 
import os
import json

# node_path = '/mydata/dgl/general/input_data_1-4/nodes/'
# edge_path = '/mydata/dgl/general/input_data_1-4/edges'

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
        df.drop_duplicates(inplace=True)
        df = downsize_data(df)
        for f in files[1:]:
            print(f"Inside second for {os.path.join(root, f)}")
            df_temp = pd.read_parquet(os.path.join(root, f))
            df_temp.drop_duplicates(inplace=True)
            df_temp = downsize_data(df_temp)
            df = pd.concat([df, df_temp], axis=0)
    return df

def print_class_map(list1, list2):
    class_map = {}
    for item1, item2 in zip(list1, list2):
        if item1 not in class_map:
            class_map[item1] = item2
    
    print("👉🏻 Maps here!!: ", class_map)

def updated_edges(sampled_nodes_data, edges_data):
    variant_origin_map = set()
    selected_edges_data = []

    print(f"👉🏻 Previous number of edges: {len(edges_data)}")

    for _, row in sampled_nodes_data.iterrows():
        if (row['origin'], row['variant']) not in variant_origin_map:
            variant_origin_map.add((row['origin'], row['variant']))

    for _, row in edges_data.iterrows():
        item1 = (row['src'], row['variant_file'])
        item2 = (row['dest'], row['variant_file'])
        if item1 in variant_origin_map or item2 in variant_origin_map:
            selected_edges_data.append(row.to_dict())

    return pd.DataFrame(selected_edges_data)

def load_genomics_graph(edge_path, node_path):
    nodes_data = concat_data(node_path)
    edges_data = concat_data(edge_path)

    # Test set
    result = edges_data.loc[edges_data['dest']==19, ['variant_file', 'src']]
    result.rename(columns={'variant_file': 'variant', 'src': 'origin'}, inplace=True)
    test_set = pd.merge(result, nodes_data, on=['variant', 'origin'], how='inner')
    test_set.drop_duplicates(subset=['variant', 'origin', 'alt_genome', 'position', 'quality'], keep=False, inplace=True)

    # Train set: Validation set is untouched since it should be a combination of the other two sets
    train_set = pd.merge(nodes_data, test_set, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    sample_length = int(len(train_set.loc[train_set['cadd_class'] == 0]))
    sampled_train_set = pd.concat([train_set.loc[train_set['cadd_class'] == 0], train_set.loc[train_set['cadd_class'] == 2248601], train_set.loc[train_set['cadd_class'] == 2248607].sample(n=sample_length), train_set.loc[train_set['cadd_class'] == 2248655].sample(n=sample_length)])

    diff_sampled_train_set = pd.concat([train_set, sampled_train_set, sampled_train_set]).drop_duplicates(keep=False)
    edges_data = updated_edges(diff_sampled_train_set, edges_data)

    print(f"👉🏻 Nodes length: {len(nodes_data)}")
    print(f"👉🏻 New number of edges: {len(edges_data)}")

    test_size = len(test_set)
    train_size = len(sampled_train_set)

    print(f"👉🏻 Test set length: {test_size}")
    print(f"👉🏻👉🏻 Test CADD Scores distribution: {test_set.groupby('cadd_class')['cadd_class'].count()}")
    print(f"👉🏻 Train set length: {train_size}")
    print(f"👉🏻👉🏻 Train CADD Scores distribution: {sampled_train_set.groupby('cadd_class')['cadd_class'].count()}")

    print(f"👉🏻 Writing sampled train set, test set and edges data...")
    edges_data.to_parquet('/mydata/dgl/general/partitioned_data/new_edges_data.parquet', index=False)
    sampled_train_set.to_parquet('/mydata/dgl/general/partitioned_data/sampled_train_set.parquet', index=False)
    test_set.to_parquet('/mydata/dgl/general/partitioned_data/test_set.parquet', index=False)

    # START: Loading existing files
    # edges_data = pd.read_parquet('/mydata/dgl/general/new_edges_data.parquet')
    # sampled_train_set = pd.read_parquet('/mydata/dgl/general/sampled_train_set.parquet')
    # test_set = pd.read_parquet('/mydata/dgl/general/test_set.parquet')

    # test_size = len(test_set)
    # train_size = len(sampled_train_set)

    # print(f"👉🏻 Test set length: {test_size}")
    # print(f"👉🏻👉🏻 Test CADD Scores distribution: {test_set.groupby('cadd_class')['cadd_class'].count()}")
    # print(f"👉🏻 Train set length: {train_size}")
    # print(f"👉🏻👉🏻 Train CADD Scores distribution: {sampled_train_set.groupby('cadd_class')['cadd_class'].count()}")
    # # END

    new_nodes = pd.concat([sampled_train_set, test_set], axis=0)
    
    node_features = torch.from_numpy(new_nodes.iloc[:, 2:-2].to_numpy(dtype='int64'))
    node_labels = torch.from_numpy(new_nodes['cadd_class'].astype('category').cat.codes.to_numpy()).to(torch.long)
    # json.dump({new_nodes['cadd_class'].astype('category').cat.codes.to_numpy()}, open())
    print_class_map(new_nodes['cadd_class'].tolist(), new_nodes['cadd_class'].astype('category').cat.codes.tolist())
    edge_features = torch.from_numpy(edges_data['predicate'].to_numpy(dtype='int64'))
    edges_src = torch.from_numpy(edges_data['src'].to_numpy(dtype='int64'))
    edges_dst = torch.from_numpy(edges_data['dest'].to_numpy(dtype='int64'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=new_nodes.shape[0])
    # graph = dgl.graph((edges_src, edges_dst), num_nodes=edges_data.src.max())
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    n_nodes = new_nodes.shape[0]
    n_train = int(train_size * 0.8)
    n_val = int(train_size * 0.2)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    # graph = dgl.add_reverse_edges(graph)
    return graph
