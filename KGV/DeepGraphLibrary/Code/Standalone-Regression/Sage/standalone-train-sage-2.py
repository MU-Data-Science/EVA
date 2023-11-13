import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import DGLDataset
import glob
import argparse
import pandas as pd
from dgl.nn import GraphConv
# from dgl.data import citation_graph as citegrh

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

# Step 1: Load the dataset
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
        print(f"length of nodes_data: {len(nodes_data)}")
        edges_data = concat_data(edge_path)[:nm]
        print(f"length of edges_data: {len(edges_data)}")

        result = edges_data.loc[edges_data['dest']==19, ['variant_file', 'src']]
        result.rename(columns={'variant_file': 'variant', 'src': 'origin'}, inplace=True)
        test_set = pd.merge(result, nodes_data, on=['variant', 'origin'], how='inner')
        test_set.drop_duplicates(subset=['variant', 'origin', 'alt_genome', 'position', 'quality'], keep=False, inplace=True)

        train_set = pd.merge(nodes_data, test_set, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

        test_size = len(test_set)
        train_size = len(train_set)

        # print(self.test_size, self.train_size)

        new_nodes = pd.concat([train_set, test_set], axis=0)

        node_features = torch.from_numpy(new_nodes.iloc[:, 2:-2].to_numpy())
        node_labels = torch.from_numpy(new_nodes['raw_scores'].to_numpy()).to(torch.float)
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

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
    
dataset = OurDataset()
# g = dataset.graph
g = dataset[0]
g = dgl.add_self_loop(g)
print(g) 
features = g.ndata['feat']
labels = g.ndata['label']

# Step 2: Define the model architecture
class GraphClassifier(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GraphClassifier, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))
        x = F.relu(self.conv2(g, x))
        x = self.fc(x)
        return x

# Step 3: Create the model instance
model = GraphClassifier(in_feats=features.shape[1], hidden_size=64, num_classes=1)

# Step 4: Define the loss function
loss_fn = nn.MSELoss()

# Step 5: Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 7: Train
def train(model, g, features, labels, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    logits = model(g, features)
    loss = loss_fn(logits.squeeze(), labels.squeeze())
    loss.backward()
    optimizer.step()
    return loss.item()

# Step 7: Evaluation
def evaluate(model, g, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        predicted_labels = logits.squeeze()
        print(f"Logits: {logits}, Predicted: {predicted_labels}")
        mse = torch.mean((predicted_labels - labels.squeeze()) ** 2)
        acc = compute_accuracy(predicted_labels, labels.squeeze())
    return mse, acc

def compute_accuracy(pred_labels, true_labels):
    # Round predicted labels to the nearest integer
    rounded_pred = torch.round(pred_labels)
    correct = torch.sum(rounded_pred == true_labels)
    acc = correct.item() / len(true_labels)
    return acc

# Training loop
for epoch in range(100):
    loss = train(model, g, features, labels, optimizer, loss_fn)
    mse, acc = evaluate(model, g, features, labels)
    print('Epoch {}, Loss: {:.4f}, MSE: {:.4f}, Accuracy: {:.2f}%'.format(epoch, loss, mse, acc * 100))

# mse = evaluate(model, g, features, labels)
# print('Mean Squared Error: {:.4f}'.format(mse))
