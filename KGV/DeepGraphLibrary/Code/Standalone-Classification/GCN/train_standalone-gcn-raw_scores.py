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

import gc

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--edge", required=True, help="edge features")
ap.add_argument("-n", "--node", required=True, help="node features")
ap.add_argument("-e", "--epoch", required=True, help="number of epochs")
ap.add_argument("-lr", "--lr", required=True, help="learning rate")

args = vars(ap.parse_args())
edge_path = str(args['edge'])
node_path = str(args['node'])
epoch = int(args['epoch'])
lr = float(args['lr'])

PATH_TO_SAVE_MODEL = '/mydata/dgl/general/Model/'
os.makedirs(PATH_TO_SAVE_MODEL, exist_ok=True)

def concat_data(filepath):
    if os.path.isfile(filepath):
        df = pd.read_parquet(filepath)
    else:
        for root, dirs, files in os.walk(filepath):
            print(f"Inside first for {os.path.join(root, files[0])}")
            df = pd.read_parquet(os.path.join(root, files[0]))
            for f in files[1:]:
                print(f"Inside second for {os.path.join(root, f)}")
                df_temp = pd.read_parquet(os.path.join(root, f))
                df = pd.concat([df, df_temp], axis=0)
    return df

print("***************Graph Creation***************")
class OurDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='our_dataset')

    def process(self):
        nodes_data = concat_data(node_path)
        print(f"length of nodes_data: {len(nodes_data)}")
        edges_data = concat_data(edge_path)
        print(f"length of edges_data: {len(edges_data)}")

        nodes_data = nodes_data.loc[nodes_data['chromosome'] != 0]

        d = {(0, 10): 1, (10, 20): 2, (20, 100): 3}
        nodes_data['cadd_class'] = nodes_data['raw_scores'].apply(lambda x: next((v for k, v in d.items() if x>=k[0] and x<k[1]), 0))

        print(nodes_data.head(10))
        print(nodes_data.groupby('cadd_class').count())
        
        node_features = torch.from_numpy(nodes_data.iloc[:, 2:-3].to_numpy())
        node_labels = torch.from_numpy(nodes_data['cadd_class'].astype('category').cat.codes.to_numpy()).to(torch.long) 
        
        # edge_features = torch.from_numpy(np.ones(edges_data.shape[0]))
        edge_features = torch.from_numpy(edges_data['predicate'].to_numpy())
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['dest'].to_numpy())
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
    
dataset = OurDataset()
g = dataset[0]
g = dgl.add_self_loop(g)

print(g)    

nodes_data = None
edges_data = None

gc.collect()
print("***************Node Prediction***************")
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


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

# model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    for e in range(epoch):
        logits = model(g, features)
        pred = logits.argmax(1)
        
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            torch.save(model.state_dict(), PATH_TO_SAVE_MODEL + f'best_weights_epoch_{e}.pt')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 1 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

model = GCN(g.ndata['feat'].shape[1], 16, 4)
train(g, model)

#To save model:
# trained_model = th.load("/mydata/dgl/")