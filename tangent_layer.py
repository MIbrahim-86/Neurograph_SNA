import torch
import numpy as np

def map_to_tangent_matrix(matrices, epsilon=1e-4):
    """
    Performs the Log-Euclidean Map manually to keep the matrix structure.
    Formula: Log(C) = U * Log(Eigenvalues) * U.T
    """
    # 1. Force Symmetry
    matrices = (matrices + np.swapaxes(matrices, 1, 2)) / 2.0
    
    # 2. Eigen Decomposition
    eigvals, eigvecs = np.linalg.eigh(matrices)
    
    # 3. Clip negative eigenvalues (Repair broken matrices)
    eigvals[eigvals < epsilon] = epsilon
    
    # 4. Apply Logarithm to Eigenvalues (The Tangent Projection)
    # This flattens the manifold curvature
    log_eigvals = np.log(eigvals)
    
    # 5. Reconstruct the Matrix (U * Log(Eig) * U.T)
    # Broadcasting: (N, D, D) * (N, D, 1) -> (N, D, D)
    reconstructed = np.matmul(eigvecs * log_eigvals[:, np.newaxis, :], np.swapaxes(eigvecs, 1, 2))
    
    return reconstructed

def project_to_tangent_matrix(loader, device):
    all_matrices = []
    all_labels = []
    
    print("Extracting and projecting matrices...")
    
    for data in loader:
        num_graphs = data.num_graphs
        num_nodes = data.x.shape[0] // num_graphs
        
        # Reshape into (Batch_Size, Num_Nodes, Num_Nodes)
        matrices = data.x.reshape(num_graphs, num_nodes, num_nodes).cpu().numpy()
        matrices = np.nan_to_num(matrices)
        
        all_matrices.append(matrices)
        all_labels.append(data.y.cpu().numpy())

    X = np.concatenate(all_matrices)
    y = np.concatenate(all_labels)

    # Transform to Tangent Matrix
    X_tangent = map_to_tangent_matrix(X)

    # Return as (N, 1000, 1000)
    return (torch.tensor(X_tangent, dtype=torch.float32).to(device), torch.tensor(y).to(device))

def get_dataloaders_tangent(train_loader, val_loader, test_loader, device):
    """
    Wrapper to process all sets
    """
    print("Processing Training Set...")
    train_data = project_to_tangent_matrix(train_loader, device)
    
    print("Processing Validation Set...")
    val_data = project_to_tangent_matrix(val_loader, device)
    
    print("Processing Test Set...")
    test_data = project_to_tangent_matrix(test_loader, device)
    
    return train_data, val_data, test_data