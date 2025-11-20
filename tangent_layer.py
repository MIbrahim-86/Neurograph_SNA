import torch
import numpy as np
from pyriemann.tangentspace import TangentSpace

def make_pd(matrices, epsilon=1e-4):
    """
    Forces matrices to be Symmetric Positive Definite (SPD) by:
    1. Symmetrizing: (A + A.T) / 2
    2. Eigen-clipping: Setting all negative eigenvalues to epsilon
    """
    # 1. Force Symmetry
    matrices = (matrices + np.swapaxes(matrices, 1, 2)) / 2.0
    
    # 2. Eigen Decomposition
    eigvals, eigvecs = np.linalg.eigh(matrices)
    
    # 3. Clip eigenvalues (Set any val < epsilon to epsilon)
    eigvals[eigvals < epsilon] = epsilon
    
    # 4. Reconstruct
    # (V * diag(vals) * V.T)
    # We use broadcasting to multiply (N, D, D) by (N, D)
    reconstructed = np.matmul(eigvecs * eigvals[:, np.newaxis, :], np.swapaxes(eigvecs, 1, 2))
    
    return reconstructed

def project_to_tangent_space(loader, device):
    """
    Takes a PyG Data Loader, extracts correlation matrices, 
    repairs them, and converts to Tangent Space.
    """
    all_matrices = []
    all_labels = []
    
    print("Extracting matrices...")
    
    for data in loader:
        num_graphs = data.num_graphs
        num_nodes = data.x.shape[0] // num_graphs
        
        # Reshape into (Batch_Size, Num_Nodes, Num_Nodes)
        matrices = data.x.reshape(num_graphs, num_nodes, num_nodes).cpu().numpy()
        
        # Check for NaNs just in case and replace with 0
        matrices = np.nan_to_num(matrices)
        
        all_matrices.append(matrices)
        all_labels.append(data.y.cpu().numpy())

    X = np.concatenate(all_matrices)
    y = np.concatenate(all_labels)

    # --- THE FIX IS HERE ---
    print(f"Repairing {len(X)} matrices to ensure Positive Definiteness...")
    X = make_pd(X)
    # -----------------------

    return X, y

def fit_transform_tangent(train_loader, val_loader, test_loader, device):
    # 1. Load and Repair Matrices
    X_train, y_train = project_to_tangent_space(train_loader, device)
    X_val, y_val = project_to_tangent_space(val_loader, device)
    X_test, y_test = project_to_tangent_space(test_loader, device)

    print(f"Fitting Tangent Space on {len(X_train)} training subjects (1000x1000)...")
    print("Using Log-Euclidean metric for speed optimization.")

    # 2. Initialize with 'logeuclid' metric (FAST)
    # n_jobs=-1 uses all CPU cores to speed up the projection
    ts = TangentSpace(metric='logeuclid') 
    
    # 3. Fit and Transform
    # This should now take minutes instead of hours
    X_train_ts = ts.fit_transform(X_train)
    X_val_ts = ts.transform(X_val)
    X_test_ts = ts.transform(X_test)

    print("Tangent Space projection complete.")
    
    return (torch.tensor(X_train_ts, dtype=torch.float32).to(device), torch.tensor(y_train).to(device)), \
           (torch.tensor(X_val_ts, dtype=torch.float32).to(device), torch.tensor(y_val).to(device)), \
           (torch.tensor(X_test_ts, dtype=torch.float32).to(device), torch.tensor(y_test).to(device))