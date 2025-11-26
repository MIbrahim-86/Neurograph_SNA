from NeuroGraph.datasets import NeuroGraphDataset
from tangent_layer import fit_transform_tangent
import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import os,random
import os.path as osp
import sys
import time
from utils import *

class TangentMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TangentMLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.BatchNorm1d(hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HCPGender')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--model', type=str, default="GCNConv")
parser.add_argument('--hidden', type=int, default=32)
parser.add_argument('--hidden_mlp', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--echo_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--early_stopping', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.5)
args = parser.parse_args()
path = "base_params/"
res_path = "results/"
root = "data/"
if not os.path.isdir(path):
    os.mkdir(path)
if not os.path.isdir(res_path):
    os.mkdir(res_path)
def logger(info):
    f = open(os.path.join(res_path, 'results_new.csv'), 'a')
    print(info, file=f)

fix_seed(args.seed)
dataset = NeuroGraphDataset(root=root, name= args.dataset)
print(dataset.num_classes)
print(len(dataset))

print("dataset loaded successfully!",args.dataset)
labels = [d.y.item() for d in dataset]

train_tmp, test_indices = train_test_split(list(range(len(labels))),
                        test_size=0.2, stratify=labels,random_state=args.seed,shuffle= True)
tmp = dataset[train_tmp]
train_labels = [d.y.item() for d in tmp]
train_indices, val_indices = train_test_split(list(range(len(train_labels))),
 test_size=0.125, stratify=train_labels,random_state=args.seed,shuffle = True)
train_dataset = tmp[train_indices]
val_dataset = tmp[val_indices]
test_dataset = dataset[test_indices]
print("dataset {} loaded with train {} val {} test {} splits".format(args.dataset,len(train_dataset), len(val_dataset), len(test_dataset)))

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False,drop_last=True)
val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False,drop_last=True)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False,drop_last=True)
args.num_features,args.num_classes = dataset.num_features,dataset.num_classes


print("Starting Tangent Space Projection...")
# This converts your graph data into Tangent Space Vectors
(train_X_ts, train_y_ts), (val_X_ts, val_y_ts), (test_X_ts, test_y_ts) = fit_transform_tangent(train_loader, val_loader, test_loader, args.device)

input_dim = train_X_ts.shape[1] # This is the size of the new feature vector
print(f"New Input Dimension (Tangent Space): {input_dim}")


criterion = torch.nn.CrossEntropyLoss()
#criterion = torch.nn.L1Loss()
def train_tangent(X, y, model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test_tangent(X, y, model):
    model.eval()
    out = model(X)
    pred = out.argmax(dim=1)
    correct = int((pred == y).sum())
    return correct / len(y)

# The Main Loop
val_acc_history, test_acc_history, test_loss_history = [], [], []
seeds = [args.seed + i for i in range(args.runs)]


for index in range(args.runs):
    print(f"--- RUN {index+1}/{args.runs} ---")
    fix_seed(seeds[index])
    
    # Initialize the TangentMLP instead of ResidualGNNs
    model = TangentMLP(input_dim, args.hidden_mlp, args.num_classes).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train on Tangent Vectors
        loss = train_tangent(train_X_ts, train_y_ts, model, optimizer)
        val_acc = test_tangent(val_X_ts, val_y_ts, model)
        test_acc = test_tangent(test_X_ts, test_y_ts, model)
        
        print("epoch: {}, loss: {}, val_acc:{}, test_acc:{}".format(epoch, np.round(loss, 6), np.round(val_acc, 2), np.round(test_acc, 2)))
        
        if val_acc >= best_val_acc: # Added >= so it saves even if equal
            best_val_acc = val_acc
            # Save immediately, no waiting for half epochs
            torch.save(model.state_dict(), path + args.dataset + 'TangentMLP' + 'task-checkpoint-best-acc.pkl')

    # Load best model and test
    model.load_state_dict(torch.load(path + args.dataset + 'TangentMLP' + 'task-checkpoint-best-acc.pkl'))
    final_test_acc = test_tangent(test_X_ts, test_y_ts, model)
    test_acc_history.append(final_test_acc)
    print(f"Run {index+1} Final Test Acc: {final_test_acc}")