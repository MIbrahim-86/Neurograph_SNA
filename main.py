from NeuroGraph.datasets import NeuroGraphDataset
import argparse
import glob
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import os
from utils import fix_seed
from tangent_layer import map_to_tangent_matrix # Ensure tangent_layer.py exists
from spdnet_layer import BiMap, ReEig, LogEig, stiefel_optimizer_step

# --- SPDNET ARCHITECTURE ---
class SPDNet(nn.Module):
    def __init__(self, in_dim, mid_dim, num_classes):
        super(SPDNet, self).__init__()
        
        # Layer 1: Compress 1000x1000 -> 64x64
        # This effectively learns the "Principal Components" on the manifold
        self.layer1 = BiMap(in_dim, mid_dim)
        self.act1 = ReEig()
        
        # Layer 2: Compress 64x64 -> 32x32 (Optional deep layer)
        self.layer2 = BiMap(mid_dim, mid_dim // 2)
        self.act2 = ReEig()
        
        # Layer 3: Flatten (Tangent Projection)
        self.log_eig = LogEig()
        
        # Layer 4: Standard Linear Classifier
        # Input size is the upper triangle of the resulting matrix
        flat_dim = ((mid_dim // 2) * ((mid_dim // 2) + 1)) // 2
        self.classifier = nn.Linear(flat_dim, num_classes)

    def forward(self, x):
        # x: (Batch, 1000, 1000)
        
        x = self.layer1(x) # -> (Batch, 64, 64)
        x = self.act1(x)
        
        x = self.layer2(x) # -> (Batch, 32, 32)
        x = self.act2(x)
        
        x = self.log_eig(x)
        
        # Vectorize (Take upper triangle only)
        # Because the result is symmetric, we only need half the values
        batch, r, c = x.shape
        idx = torch.triu_indices(r, c)
        x_flat = x[:, idx[0], idx[1]]
        
        return self.classifier(x_flat)
    
    
    
def save_if_new_record(model, dataset_name, current_acc):
    # 1. Search for existing best files for this dataset
    # Pattern: "spdnet_best_HCPAge_*.pkl"
    existing_files = glob.glob(f"spdnet_best_{dataset_name}_*.pkl")
    
    previous_record = 0.0
    file_to_delete = None

    if existing_files:
        # 2. Extract the accuracy from the filename
        # Example: "spdnet_best_HCPAge_0.6021.pkl" -> 0.6021
        for f in existing_files:
            try:
                # Split by '_' and take the last part, remove .pkl
                acc_str = f.split('_')[-1].replace('.pkl', '')
                acc = float(acc_str)
                
                if acc > previous_record:
                    previous_record = acc
                    file_to_delete = f
            except:
                pass # Ignore badly named files

    # 3. Compare: Is the new one better?
    if current_acc > previous_record:
        print(f"🏆 NEW RECORD! ({current_acc:.4f} > {previous_record:.4f}). Saving model...")
        
        # Delete the old weak model
        if file_to_delete:
            os.remove(file_to_delete)
        
        # Save the new champion
        torch.save(model.state_dict(), f"spdnet_best_{dataset_name}_{current_acc:.4f}.pkl")
    else:
        print(f"❌ Current ({current_acc:.4f}) did not beat record ({previous_record:.4f}). Discarding.")

# --- HELPER: GET RAW MATRICES (NO LOG) ---
def get_raw_matrices(loader, device):
    all_matrices = []
    all_labels = []
    
    print("Loading data to GPU for preprocessing...")
    
    for data in loader:
        # 1. Keep data as PyTorch Tensors and move to GPU immediately
        num_graphs = data.num_graphs
        num_nodes = data.x.shape[0] // num_graphs
        
        # Reshape: (Batch, 1000, 1000)
        matrices = data.x.reshape(num_graphs, num_nodes, num_nodes).to(device)
        labels = data.y.to(device)
        
        # 2. Symmetrize (on GPU)
        # (A + A.T) / 2
        matrices = (matrices + matrices.transpose(1, 2)) / 2.0
        
        # 3. Eigen Decomposition (on GPU)
        # torch.linalg.eigh is MUCH faster than np.linalg.eigh
        L, V = torch.linalg.eigh(matrices)
        
        # 4. Clip/Repair Eigenvalues
        L = torch.clamp(L, min=1e-4)
        
        # 5. Reconstruct
        # V @ diag(L) @ V.T
        rec = torch.matmul(V * L.unsqueeze(1), V.transpose(-2, -1))
        
        all_matrices.append(rec)
        all_labels.append(labels)
    
    # Concatenate all batches
    X = torch.cat(all_matrices, dim=0)
    y = torch.cat(all_labels, dim=0)
    
    print(f"Loaded {X.shape[0]} matrices on {device}.")
    return X, y

# --- SETUP ---
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='HCPGender')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--mid_dim', type=int, default=48) # Target dimension for BiMap
args = parser.parse_args()

fix_seed(123)
dataset = NeuroGraphDataset(root="data/", name=args.dataset)
# ... [Split code same as before] ...
# (Copy the train/test/val splitting code from your previous main.py)
labels = [d.y.item() for d in dataset]
train_tmp, test_indices = train_test_split(list(range(len(labels))), test_size=0.2, stratify=labels, random_state=123, shuffle=True)
tmp = dataset[train_tmp]
train_labels = [d.y.item() for d in tmp]
train_indices, val_indices = train_test_split(list(range(len(train_labels))), test_size=0.125, stratify=train_labels, random_state=123, shuffle=True)
train_loader = DataLoader(tmp[train_indices], args.batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(tmp[val_indices], args.batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(dataset[test_indices], args.batch_size, shuffle=False, drop_last=True)

# Load Data (Raw SPD Matrices)
train_X, train_y = get_raw_matrices(train_loader, args.device)
val_X, val_y = get_raw_matrices(val_loader, args.device)
test_X, test_y = get_raw_matrices(test_loader, args.device)

# Initialize
model = SPDNet(in_dim=train_X.shape[1], mid_dim=args.mid_dim, num_classes=dataset.num_classes).to(args.device)
optimizer = Adam(model.parameters(), lr=args.lr) # SPDNet often needs higher LR
criterion = nn.CrossEntropyLoss()

# --- TRAINING LOOP ---
best_acc = 0.0
for epoch in range(args.epochs):
    model.train()
    indices = torch.randperm(train_X.size(0))
    total_loss = 0
    
    for i in range(0, train_X.size(0), args.batch_size):
        idx = indices[i:i+args.batch_size]
        batch_X, batch_y = train_X[idx], train_y[idx]
        
        optimizer.zero_grad()
        out = model(batch_X)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        
        # CRITICAL: Manifold Correction
        stiefel_optimizer_step(model)
        
        total_loss += loss.item()
        
    # Validation
    model.eval()
    with torch.no_grad():
        out = model(val_X)
        pred = out.argmax(dim=1)
        val_acc = int((pred == val_y).sum()) / len(val_y)
        
        test_out = model(test_X)
        test_pred = test_out.argmax(dim=1)
        test_acc = int((test_pred == test_y).sum()) / len(test_y)
    
    print(f"Epoch {epoch}: Loss {total_loss:.6f}, Val {val_acc:.6f}, Test {test_acc:.6f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        # We don't save to disk yet, we just remember it for this session.
        # We will check against the All-Time Record at the very end.
        best_model_state = model.state_dict() # Keep a copy in RAM

# --- END OF LOOP ---
print(f"Final Best Session Accuracy: {best_acc:.6f}")

if best_acc > 0:
    model.load_state_dict(best_model_state)
    save_if_new_record(model, args.dataset, best_acc)

print(f"Final Best Val Accuracy: {best_acc:.6f}")