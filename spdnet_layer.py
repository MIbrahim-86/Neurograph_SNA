import torch
import torch.nn as nn
from torch.autograd import Function

# 1. BiMap Layer (Bilinear Mapping)
# Concept: Like a Linear Layer (Matrix Multiplication), but preserves the matrix structure.
# Math: X_new = W.T @ X @ W
class BiMap(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BiMap, self).__init__()
        # Initialize weights as semi-orthogonal (Stiefel Manifold)
        # This ensures we rotate the "globe" without stretching it weirdly
        self.W = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        nn.init.orthogonal_(self.W)

    def forward(self, x):
        # x: (Batch, N, N)
        # Result: (Batch, out_dim, out_dim)
        return torch.matmul(torch.matmul(self.W.t(), x), self.W)

# 2. ReEig Layer (Rectified Eigenvalues)
# Concept: The "ReLU" of Manifold Networks.
# It filters out weak signals (small eigenvalues) to remove noise.
class ReEig(nn.Module):
    def __init__(self, epsilon=1e-4):
        super(ReEig, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # Eigen Decomposition
        # We use standard PyTorch eigh (batch-supported)
        eigvals, eigvecs = torch.linalg.eigh(x)
        
        # Rectify: Threshold eigenvalues (max(val, epsilon))
        eigvals = torch.clamp(eigvals, min=self.epsilon)
        
        # Reconstruct: U * Thresholded_Sigma * U.T
        # Broadcasting logic to multiply correct dims
        rec = torch.matmul(eigvecs * eigvals.unsqueeze(1), eigvecs.transpose(-2, -1))
        return rec

# 3. LogEig Layer (Log-Euclidean Projection)
# Concept: This is the "Flattening" step. 
# We do this ONLY at the very end, after the BiMap has optimized the view.
class LogEig(nn.Module):
    def __init__(self):
        super(LogEig, self).__init__()

    def forward(self, x):
        eigvals, eigvecs = torch.linalg.eigh(x)
        
        # Logarithm of eigenvalues
        # (This linearizes the curved distances)
        eigvals = torch.log(torch.clamp(eigvals, min=1e-6))
        
        # Reconstruct
        out = torch.matmul(eigvecs * eigvals.unsqueeze(1), eigvecs.transpose(-2, -1))
        return out

# 4. The Stiefel Optimizer Hook
# CRITICAL: Standard SGD/Adam ruins the geometry of the BiMap weights.
# We must "fix" the weights after every update to keep them orthogonal.
def stiefel_optimizer_step(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, BiMap):
                # QR Decomposition to re-orthogonalize W
                Q, R = torch.linalg.qr(m.W)
                # Ensure diagonal of R is positive for uniqueness
                sign_diag = torch.sign(torch.diag(R))
                m.W.data = Q * sign_diag