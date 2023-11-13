# pip3 install torch-geometric

import torch
torch.manual_seed(0)
import random
random.seed(0)

from create_graph_for_ann_effect import read_from_raw_data

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
ap.add_argument("-g", "--graph_input", required=False, help="input graph")
ap.add_argument("-d", "--edge", required=True, help="edge features")
ap.add_argument("-n", "--node", required=True, help="node features")
ap.add_argument("-e", "--epoch", required=True, help="number of epochs")
ap.add_argument("-lr", "--lr", required=True, help="learning rate")
ap.add_argument("-hl", "--hl", required=True, help="learning rate")

args = vars(ap.parse_args())
edge_path = str(args['edge'])
node_path = str(args['node'])
graph_path = str(args['graph_input']) if not None else None
epoch = int(args['epoch'])
lr = float(args['lr'])
hidden_layers = int(args['hl'])

b = str(lr).split('.')[1]
PATH_TO_SAVE_MODEL = f'/mydata/dgl/general/Model-SAGE-{hidden_layers}-{b}/'
os.makedirs(PATH_TO_SAVE_MODEL, exist_ok=True)

print("***************Graph Creation***************")
class OurDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='our_dataset')

    def process(self):

        nodes_data, edges_data, total_selected_len = read_from_raw_data(node_path, edge_path)

        print(nodes_data.head(10))
        
        node_features_df = nodes_data.drop(['origin', 'ann_impact', 'phred_score', 'raw_scores'], axis=1)
        node_features_df['accession_id'] = nodes_data['accession_id'].astype('category').cat.codes

        node_features = torch.from_numpy(node_features_df.iloc[:, 0:-1].to_numpy())
        node_labels = torch.from_numpy(nodes_data['ann_impact'].astype('category').cat.codes.to_numpy()).to(torch.long)
        
        edge_features = torch.from_numpy(np.ones(edges_data.shape[0]))
        # edge_features = torch.from_numpy(edges_data['predicate'].to_numpy())
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['dest'].to_numpy())
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

        n_nodes = nodes_data.shape[0]
        n_train = int(total_selected_len * 0.6)
        n_val = int(total_selected_len * 0.2)
        print("#"*50, n_train, n_val, total_selected_len - (n_train + n_val))

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:total_selected_len] = True
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

graph_labels = {"glabel": torch.tensor([0, 5])}
save_graphs('/mydata/dgl/general/genomics_graph_for_ann_effect.bin', g, graph_labels)

# dataset, graph_labels = load_graphs(graph_path)
# labels = graph_labels['glabel']
# g = dataset[0]
# g = dgl.add_self_loop(g)

print(g)
print(g.is_homogeneous)

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

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat'].float()
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
            pickle.dump({'gold': labels.masked_select(test_mask), 'pred': pred.masked_select(test_mask)}, open(f"preds.pkl", 'wb'))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 1 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(e, loss, val_acc, best_val_acc, test_acc, best_test_acc))

    print(f"Completed training for hidden layers {hidden_layers} and learning rate {lr} over {epoch} epochs")

# Instantiate SAGEConv model
model = GraphSAGE(g.ndata['feat'].shape[1], hidden_layers, 5)

# Call model training function
train(g, model)
