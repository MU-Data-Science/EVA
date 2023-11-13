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
import pickle

from dgl.data.utils import save_graphs, load_graphs

import gc

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--graph_path", required=True, help="input graph path")
ap.add_argument("-m", "--model_path", required=True, help="GCN model path to be tested")
ap.add_argument("-e", "--epoch", required=True, help="number of epochs")
ap.add_argument("-lr", "--lr", required=True, help="learning rate")
ap.add_argument("-hl", "--hl", required=True, help="learning rate")

args = vars(ap.parse_args())
graph_path = str(args['graph_path'])
model_path = str(args['model_path'])
epoch = int(args['epoch'])
lr = float(args['lr'])
hidden_layers = int(args['hl'])

dataset, graph_labels = load_graphs(graph_path)
labels = graph_labels['glabel']
g = dataset[0]

print(g)    

gc.collect()

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

def evaluate_model(g, trained_model):

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    with torch.no_grad():
        model.eval()

        logits = trained_model(g, features)
        pred = logits.argmax(1)

        pickle.dump({'gold': labels.masked_select(test_mask), 'pred': pred.masked_select(test_mask)}, open(f"preds.pkl", 'wb'))

        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        print(test_acc)

model = GCN(g.ndata['feat'].shape[1], hidden_layers, 5)

trained_model = torch.load(model_path)
model.load_state_dict(trained_model)
evaluate_model(g, model)

