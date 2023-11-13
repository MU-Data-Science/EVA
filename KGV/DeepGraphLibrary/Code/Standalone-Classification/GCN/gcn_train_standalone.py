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

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--graph_input", required=True, help="input graph")
ap.add_argument("-e", "--epoch", required=True, help="number of epochs")
ap.add_argument("-lr", "--lr", required=True, help="learning rate")
ap.add_argument("-hl", "--hl", required=True, help="learning rate")

args = vars(ap.parse_args())
graph_path = str(args['graph_input'])
epoch = int(args['epoch'])
lr = float(args['lr'])
hidden_layers = int(args['hl'])

b = str(lr).split('.')[1]
PATH_TO_SAVE_MODEL = f'/mydata/dgl/general/Model-GCN-{hidden_layers}-{b}/'
os.makedirs(PATH_TO_SAVE_MODEL, exist_ok=True)

print("***************Graph Creation***************")
dataset, graph_labels = load_graphs(graph_path)
labels = graph_labels['glabel']
g = dataset[0]
g = dgl.add_self_loop(g)

print(g)

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

    print(f"Completed training for hidden layers {hidden_layers} and learning rate {lr} over {epoch} epochs")


# Instantiate GCN model
model = GCN(g.ndata['feat'].shape[1], hidden_layers, 5)

# Call model training function
train(g, model)
