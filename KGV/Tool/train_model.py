''' Step 4: Code in progress. Refer to DeepGraphLibrary folder to train GNNs.
'''
import torch
torch.manual_seed(0)
import random
random.seed(0)

import dgl
import pandas as pd
import glob
import matplotlib.pyplot as plt
import torch
import networkx as nx
import os
from dgl.data import DGLDataset
import numpy as np
import argparse
from dgl.data.utils import save_graphs, load_graphs
import gc

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

from create_graph import create_graph

nodes_path = '/mydata/dgl/general/Dataset/nodes'
edges_path = '/mydata/dgl/general/Dataset/edges'
target_col_name = 'cadd_class'
col_to_cat = 'raw_scores'
num_of_bins = 5
ratios = []
do_save = True

epoch = 1000
lr = 0.01
hidden_layers = 16

# g = create_graph(nodes_path, edges_path, target_col_name, col_to_cat, num_of_bins, ratios, do_save)

dataset, graph_labels = load_graphs('/mydata/dgl/general/new_genomics_graph.bin')
labels = graph_labels['glabel']
g = dataset[0]
g = dgl.add_self_loop(g)

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    
    for e in range(epoch):
        logits = model(g, features)
        pred = logits.argmax(1)
        
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        # test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            # best_test_acc = test_acc
            # torch.save(model.state_dict(), PATH_TO_SAVE_MODEL + f'best_weights_epoch_{e}.pt')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 1 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f})'.format(e, loss, val_acc, best_val_acc))

    print(f"Completed training for hidden layers {hidden_layers} and learning rate {lr} over {epoch} epochs")


# Instantiate GCN model
model = GCN(g.ndata['feat'].shape[1], hidden_layers, num_of_bins+1)

# Call model training function
train(g, model)
