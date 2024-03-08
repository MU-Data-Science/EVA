''' Step 3: Create a graph with the dataset. If user has requested to save data, save train, val, test splits as .parquet files and graph in .bin format.
'''

import torch
torch.manual_seed(0)
import random
random.seed(0)

import pandas as pd
import numpy as np
import os
import torch

import dgl
from dgl.data.utils import save_graphs, load_graphs

from create_class_label import create_class_label

def create_graph(nodes_path, edges_path, target_col_name, col_to_cat, num_of_bins, ratios, do_save):

    nodes_data, edges_data = create_class_label(nodes_path, edges_path, target_col_name, col_to_cat, num_of_bins)

    if not ratios:
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 1.0 - (train_ratio + val_ratio)
    else :
        train_ratio = ratios[0]
        val_ratio = ratios[1]
        test_ratio = 1.0 - (train_ratio + val_ratio)

    nodes_data.drop('ann_impact', axis=1, inplace=True)
    node_features = torch.from_numpy(nodes_data.iloc[:, 2:-4].to_numpy(dtype='int64'))
    node_labels = torch.from_numpy(nodes_data['cadd_class'].to_numpy()).to(torch.long)
    # json.dump({new_nodes['cadd_class'].astype('category').cat.codes.to_numpy()}, open())

    edge_features = torch.from_numpy(edges_data['predicate'].to_numpy(dtype='int64'))
    edges_src = torch.from_numpy(edges_data['src'].to_numpy(dtype='int64'))
    edges_dst = torch.from_numpy(edges_data['dest'].to_numpy(dtype='int64'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
    # graph = dgl.graph((edges_src, edges_dst), num_nodes=edges_data.src.max())
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    n_nodes = nodes_data.shape[0]
    n_train = int(n_nodes * train_ratio)
    n_val = int(n_nodes * val_ratio)
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
    # import pdb; pdb.set_trace();

    if do_save:
        nodes_data[:n_train].to_parquet(f'/mydata/dgl/general/new_genomic_train_set.parquet')
        nodes_data[n_train:n_train+n_val].to_parquet(f'/mydata/dgl/general/new_genomic_val_set.parquet')
        nodes_data[n_train+n_val:].to_parquet(f'/mydata/dgl/general/new_genomic_test_set.parquet')

        graph_labels = {"glabel": torch.tensor(list(range(1, num_of_bins+1)))}
        save_graphs('/mydata/dgl/general/new_genomics_graph.bin', graph, graph_labels)

    return graph
