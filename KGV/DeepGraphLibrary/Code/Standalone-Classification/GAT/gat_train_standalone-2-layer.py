import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data.utils import save_graphs, load_graphs

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            dglnn.GATConv(
                hid_size * heads[0],
                out_size,
                heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # training loop
    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--graph_input", required=True, help="input graph")
    ap.add_argument("-e", "--epoch", required=True, help="number of epochs")
    ap.add_argument("-lr", "--lr", required=True, help="learning rate")
    ap.add_argument("-dt", "--dt", default="float", help="data type(float, bfloat16)")

    args = vars(ap.parse_args())
    graph_path = str(args['graph_input'])
    epochs = int(args['epoch'])
    lr = float(args['lr'])
    dt = str(args['dt'])

    b = str(lr).split('.')[1]
    PATH_TO_SAVE_MODEL = f'/mydata/dgl/general/Model-GAT-{b}/'
    # os.makedirs(PATH_TO_SAVE_MODEL, exist_ok=True)

    print(f"Training with DGL built-in GATConv module. Re: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/train.py")

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    data, graphlabels = load_graphs(graph_path)
    
    g = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # g = g.int().to(device)
    features = g.ndata["feat"].float()
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"].bool(), g.ndata["val_mask"].bool(), g.ndata["test_mask"].bool()

    # create GAT model
    in_size = features.shape[1]
    # out_size = data.num_classes
    out_size = g.ndata["feat"].shape[1]
    model = GAT(in_size, 8, out_size, heads=[8, 1]).to(device)

    # convert model and graph to bfloat16 if needed
    if dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    # model training
    print("Training...")
    train(g, features, labels, masks, model)

    # test the model
    print("Testing...")
    acc = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))


