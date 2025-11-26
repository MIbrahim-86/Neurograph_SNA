from NeuroGraph.datasets import NeuroGraphDataset
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import os, random, time
from utils import fix_seed
from tangent_layer import get_dataloaders_tangent # New Import

# --- BRAIN NETWORK TRANSFORMER ---
class BrainTransformer(nn.Module):
    def __init__(self, num_nodes, d_model, nhead, num_layers, num_classes):
        super(BrainTransformer, self).__init__()
        
        # 1. Embedding
        self.embedding = nn.Linear(num_nodes, d_model)
        
        # 2. The "Manager" (CLS) Token
        # A learnable vector that will aggregate the brain's info
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 3. Positional Encoding
        # We add +1 to num_nodes to account for the CLS token
        self.pos_encoder = nn.Parameter(torch.randn(1, num_nodes + 1, d_model))
        
        # 4. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, # Standard expansion
            dropout=0.5,               # Higher dropout to prevent overfitting
            batch_first=True,
            norm_first=True            # Pre-Norm helps convergence
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 1000, 1000)
        
        # 1. Embed features -> (Batch, 1000, d_model)
        x = self.embedding(x)
        
        # 2. Add the Manager (CLS) Token to the front of every brain
        # Shape becomes (Batch, 1001, d_model)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. Add Positional Encoding
        x = x + self.pos_encoder
        
        # 4. Attention Mechanism
        x = self.transformer_encoder(x)
        
        # 5. Intelligent Aggregation
        # Instead of averaging everything, we only look at the Manager (Index 0)
        cls_output = x[:, 0, :]
        
        return self.classifier(cls_output)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HCPGender')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0001) # Lower LR for Transformers
parser.add_argument('--weight_decay', type=float, default=0.0001)
# Transformer Params
parser.add_argument('--d_model', type=int, default=64) # Hidden dimension
parser.add_argument('--nhead', type=int, default=4)    # Number of attention heads
parser.add_argument('--num_layers', type=int, default=2) # Number of transformer layers

args = parser.parse_args()
path = "base_params/"
root = "data/"
if not os.path.isdir(path): os.mkdir(path)

fix_seed(args.seed)
dataset = NeuroGraphDataset(root=root, name=args.dataset)
args.num_classes = dataset.num_classes
labels = [d.y.item() for d in dataset]

train_tmp, test_indices = train_test_split(list(range(len(labels))), test_size=0.2, stratify=labels, random_state=args.seed, shuffle=True)
tmp = dataset[train_tmp]
train_labels = [d.y.item() for d in tmp]
train_indices, val_indices = train_test_split(list(range(len(train_labels))), test_size=0.125, stratify=train_labels, random_state=args.seed, shuffle=True)

train_dataset = tmp[train_indices]
val_dataset = tmp[val_indices]
test_dataset = dataset[test_indices]

train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, drop_last=True)

# --- NEW DATA PREP ---
(train_X, train_y), (val_X, val_y), (test_X, test_y) = get_dataloaders_tangent(train_loader, val_loader, test_loader, args.device)
num_nodes = train_X.shape[1]
print(f"Transformer Input Shape: {train_X.shape} (Batch, Nodes, Features)")
# ---------------------

criterion = torch.nn.CrossEntropyLoss()

def train_transformer(X, y, model, optimizer, batch_size):
    model.train()
    total_loss = 0
    indices = torch.randperm(X.size(0))
    
    # Mixup Parameters
    alpha = 1.0 
    use_mixup = True # Turn this on!

    for i in range(0, X.size(0), batch_size):
        idx = indices[i:i+batch_size]
        batch_X, batch_y = X[idx], y[idx]
        
        # --- MIXUP LOGIC START ---
        if use_mixup and batch_X.size(0) > 1:
            # 1. Generate mixing ratio (lambda) from Beta distribution
            lam = np.random.beta(alpha, alpha)
            
            # 2. Shuffle the batch to get "Partner" brains
            rand_idx = torch.randperm(batch_X.size(0))
            batch_X_partner = batch_X[rand_idx]
            batch_y_partner = batch_y[rand_idx]
            
            # 3. Create the "Hybrid" Brain (Linear interpolation in Tangent Space)
            mixed_X = lam * batch_X + (1 - lam) * batch_X_partner
            
            # 4. Forward pass with Hybrid Brain
            optimizer.zero_grad()
            out = mixed_X # Just a placeholder variable name
            out = model(mixed_X)
            
            # 5. Calculate "Mixed" Loss
            # Loss = lam * Loss(A) + (1-lam) * Loss(B)
            loss = lam * criterion(out, batch_y) + (1 - lam) * criterion(out, batch_y_partner)
        else:
            # Standard training if batch is too small or mixup is off
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            
        # --- MIXUP LOGIC END ---

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / (X.size(0) / batch_size)
@torch.no_grad()
def test_transformer(X, y, model, batch_size):
    model.eval()
    correct = 0
    for i in range(0, X.size(0), batch_size):
        # No shuffling for testing
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        
        out = model(batch_X)
        pred = out.argmax(dim=1)
        correct += int((pred == batch_y).sum())
    return correct / len(y)

# ... [After the test_transformer function] ...
seeds = [args.seed + i for i in range(args.runs)]
for index in range(args.runs):
    print(f"--- RUN {index+1}/{args.runs} ---")
    fix_seed(seeds[index])
    
    model = BrainTransformer(
        num_nodes=num_nodes, 
        d_model=args.d_model, # Keeping your original size
        nhead=4,              # Keeping original heads
        num_layers=2,         # Keeping original layers
        num_classes=args.num_classes
    ).to(args.device)
    
    # UPGRADE 1: Cosine Scheduler
    # Starts at lr=0.0005 and lowers it to 0 by the end
    optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # UPGRADE 2: Label Smoothing
    # Prevents the model from being "too confident" (Overfitting)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train
        loss = train_transformer(train_X, train_y, model, optimizer, args.batch_size)
        
        # Step the scheduler
        scheduler.step()
        
        # Test
        val_acc = test_transformer(val_X, val_y, model, args.batch_size)
        test_acc = test_transformer(test_X, test_y, model, args.batch_size)
        
        # Print current LR to verify scheduler is working
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch: {epoch}, Loss: {loss:.4f}, LR: {current_lr:.6f}, Val: {val_acc:.2f}, Test: {test_acc:.2f}")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), path + args.dataset + 'Transformer' + 'best.pkl')

    # Final Check
    print(f"Best Validation Accuracy: {best_val_acc:.2f}")