import torch
torch.manual_seed(0)
import random
random.seed(0)

import dgl
from dgl.data.utils import save_graphs, load_graphs
import torch_geometric.transforms as T

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data import DGLDataset
import pandas as pd
import os

import numpy as np
import pandas as pd
import pickle

from dgl.nn import GraphConv
from dgl.nn import SAGEConv
dgl.use_libxsmm(False)
print(dgl.__version__)

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--graph_path", required=True, help="input graph path")
ap.add_argument("-m", "--model_path", required=True, help="GCN model path to be tested")
ap.add_argument("-hl", "--hl", required=True, help="learning rate")

args = vars(ap.parse_args())
graph_path = str(args['graph_path'])
model_path = str(args['model_path'])
hidden_layers = int(args['hl'])

dataset, graph_labels = load_graphs(graph_path)
labels = graph_labels['glabel']
g = dataset[0]

print(g)    
import torch.nn.functional as F
from torch.nn import Linear, Dropout

class GraphSAGE(torch.nn.Module):
    """GraphSAGE"""
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(in_feats, h_feats, 'mean')
        self.sage2 = SAGEConv(h_feats, h_feats, 'mean')
        self.fc = nn.Linear(h_feats, num_classes)

    def forward(self, g, in_feat):
        x = F.relu(self.sage1(g, in_feat))
        x = F.relu(self.sage2(g, x))
        x = self.fc(x)
        return x

def evaluate_model(g, trained_model):

    features = g.ndata['feat'].float()
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    # import pdb; pdb.set_trace();
    with torch.no_grad():
        model.eval()

        logits = trained_model(g, features)
        pred = logits.argmax(1)

        pickle.dump({'gold': labels.masked_select(test_mask), 'pred': pred.masked_select(test_mask)}, open(f"preds_{hidden_layers}.pkl", 'wb'))

        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        print(test_acc)

# Instantiate SAGEConv model
model = GraphSAGE(g.ndata['feat'].shape[1], hidden_layers, 5)

trained_model = torch.load(model_path)
model.load_state_dict(trained_model)
evaluate_model(g, model)
