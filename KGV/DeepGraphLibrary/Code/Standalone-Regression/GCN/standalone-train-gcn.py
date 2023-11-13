import torch
torch.manual_seed(0)

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
import torch.nn.functional as F
from sklearn.metrics import r2_score

import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--edge", required=True, help="edge features")
ap.add_argument("-n", "--node", required=True, help="node features")
ap.add_argument("-e", "--epoch", required=True, help="number of epochs")
ap.add_argument("-m", "--model_path", required=True, help="model path")
ap.add_argument("-nm", "--number", required=False, help="number")

args = vars(ap.parse_args())
edge_path = str(args['edge'])
node_path = str(args['node'])
epoch = int(args['epoch'])
nm = int(args['number'])
PATH_TO_SAVE_MODEL = str(args['model_path']) 

def concat_data(filepath):
    files = glob.glob(filepath)
    df = pd.concat([pd.read_parquet(f) for f in files], axis=0)
    return df

print("***************Graph Creation***************")
class OurDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='our_dataset')

    def process(self): 
        nodes_data = concat_data(node_path)[:nm]
        edges_data = concat_data(edge_path)[:nm]

        result = edges_data.loc[edges_data['dest']==19, ['variant_file', 'src']]
        result.rename(columns={'variant_file': 'variant', 'src': 'origin'}, inplace=True)
        test_set = pd.merge(result, nodes_data, on=['variant', 'origin'], how='inner')
        print(f"👉🏻 Test set length (before): {len(test_set)}")
        test_set.drop(['variant', 'origin'], axis=1, inplace=True)
        test_set.drop_duplicates(subset=['alt_genome', 'position', 'quality'], keep=False, inplace=True)

        train_set = pd.merge(nodes_data, test_set, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
        print(f"👉🏻 Train set length (before): {len(train_set)}")
        train_set.drop(['variant', 'origin'], axis=1, inplace=True)
        train_set.drop_duplicates(inplace=True)

        test_size = len(test_set)
        train_size = len(train_set) 

        # Getting final information
        print(f"Length of nodes_data: {len(nodes_data)}")
        print(f"Length of edges_data: {len(edges_data)}")
        print(f"👉🏻 Test set length (after): {test_size}")
        # print(f"👉🏻👉🏻 Test CADD Scores distribution: {test_set.groupby('cadd_class_using_phred')['cadd_class_using_phred'].count()}")
        print(f"👉🏻 Train set length (after): {train_size}")
        # print(f"👉🏻👉🏻 Train CADD Scores distribution: {train_set.groupby('cadd_class_using_phred')['cadd_class_using_phred'].count()}")

        # print(self.test_size, self.train_size)

        new_nodes = pd.concat([train_set, test_set], axis=0)
        # new_nodes[:train_size].to_csv('/mydata/check.csv')

        node_features = torch.from_numpy(new_nodes.iloc[:, 2:-2].to_numpy())
        node_labels = torch.from_numpy(new_nodes['raw_scores'].to_numpy()).to(torch.float)
        print("node_labels", node_labels)
        edge_features = torch.from_numpy(edges_data['predicate'].to_numpy())
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['dest'].to_numpy())

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=new_nodes.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

        n_nodes = new_nodes.shape[0]
        n_train = int(train_size * 0.8)
        n_val = int(train_size * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        print(f"Test set size: {n_nodes - (n_train + n_val)}")
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        # print("val_labels", torch.masked_select(self.graph.ndata['label'], val_mask))

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
    
dataset = OurDataset()
g = dataset[0]
# g = dgl.add_self_loop(g)
print(g)    

print("***************Node Regression***************")
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

def plot_losses(history):

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(y=history['train_loss'], name="train loss"),secondary_y=True,)
    fig.add_trace(go.Scatter(y=history['val_loss'], name="val loss"), secondary_y=True,)
    fig.add_trace(go.Scatter(y=history['test_loss'], name="test loss"), secondary_y=True,)

    # Add figure title
    fig.update_layout(title_text="Losses")

    # Set x-axis title
    fig.update_xaxes(title_text="Epoch")
    fig.write_image("/mydata/dgl/general/fig1.png")

    # print(epoch, mse_train_loss, val_loss, test_loss)


val_df = pd.DataFrame()
test_df = pd.DataFrame()

def train(g, model):
    history = {'e': [], 'train_loss': [], 'val_loss': [], 'test_loss': []}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    best_val_loss = -1
    best_test_loss = 0

    features = g.ndata['feat']
    # import pdb; pdb.set_trace()
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    val_size = torch.sum(val_mask.to(torch.int))
    test_size = torch.sum(test_mask.to(torch.int))
    
    for e in range(epoch):
        logits = model(g, features)
        squeezed_logits = logits.squeeze(dim=1)

        val_logits = torch.masked_select(squeezed_logits, val_mask)
        val_labels = torch.masked_select(labels, val_mask)
        test_logits = torch.masked_select(squeezed_logits, test_mask)
        test_labels = torch.masked_select(labels, test_mask)

        loss = F.mse_loss(logits[train_mask], labels[train_mask].unsqueeze(1))
        val_loss = F.mse_loss(val_logits, val_labels)
        test_loss = F.mse_loss(test_logits, test_labels)

        # tot = ((labels[test_mask] - labels[test_mask].mean()) ** 2).sum()
        # res = ((labels[test_mask] - logits[test_mask]) ** 2).sum()
        # r2 = 1 - res / tot

        if best_val_loss > val_loss or best_val_loss == -1:
            best_val_loss = val_loss
            best_test_loss = test_loss
            torch.save(model.state_dict(), PATH_TO_SAVE_MODEL + f'/best_weights_epoch_{e}.pt')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 1 == 0:
            # r2 = r2_score(labels[test_mask], logits[test_mask].detach().numpy())
            print('In epoch {}, loss: {:.3f}, val loss: {:.3f} (best {:.3f}), test loss: {:.3f} (best {:.3f})'.format(e, loss, val_loss, best_val_loss, test_loss, best_test_loss))
            
            print("\t")
            print(f"Label: {val_labels}, Predicted: {val_logits}")

            val_df['label'] = val_labels.to(torch.float).tolist()
            val_df['predicted'] = val_logits.to(torch.float).tolist()
            val_df.to_csv(f'val_predictions_{time.time()}.csv')

            print(f"Val: {(torch.sum(torch.isclose(val_logits, val_labels, atol=1e-1).to(torch.int)))}, Val size: {val_size}, Val Acc: {(torch.sum(torch.isclose(val_logits, val_labels, atol=1e-1).to(torch.int)))/val_size}")

            print("\t")
            print(f"Label: {test_labels}, Predicted: {test_logits}")

            test_df['label'] = test_labels.tolist()
            test_df['predicted'] = test_logits.tolist()
            test_df.to_csv(f'test_predictions_{time.time()}.csv')

            print(f"Test: {(torch.sum(torch.isclose(test_logits, test_labels, atol=1e-1).to(torch.int)))}, Test size: {test_size}, Test Acc: {(torch.sum(torch.isclose(test_logits, test_labels, atol=1e-1).to(torch.int)))/test_size}")

            history['e'].append(e)
            history['train_loss'].append(loss.data.numpy())
            history['val_loss'].append(val_loss.data.numpy())
            history['test_loss'].append(test_loss.data.numpy())

        # if e % 5 == 0:
        #     print('In epoch {}, logits: {}, ground truth: {}'.format(e, logits, labels))

    plot_losses(history)

model = GCN(g.ndata['feat'].shape[1], 16, 1)
# train(g, model)

#To save model:
# trained_model = th.load("/mydata/dgl/general/Model/best_weights_epoch_225.pt")

