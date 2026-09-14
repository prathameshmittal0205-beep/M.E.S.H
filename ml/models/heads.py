import torch
import torch.nn as nn
import torch.nn.functional as F

class RULHead(nn.Module):
    """
    RUL Regression head.
    Outputs mean and variance to capture aleatoric uncertainty.
    """
    def __init__(self, in_features: int, hidden_dim: int, dropout_p: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        
        # We output 2 values: mean and log_variance
        self.fc2 = nn.Linear(hidden_dim, 2)
        
    def forward(self, x: torch.Tensor):
        # x shape: [batch_size, in_features]
        h = self.fc1(x)
        h = self.relu(h)
        h = self.dropout(h)
        
        out = self.fc2(h)
        mean = out[:, 0:1]
        log_var = out[:, 1:2]
        
        # Clamp log_var to prevent extreme variance values
        log_var = torch.clamp(log_var, min=-1.0, max=5.0)
        
        # Enforce positive variance (exp or softplus, using softplus for stability)
        variance = F.softplus(log_var) + 1e-6
        return mean, variance


class FaultClassificationHead(nn.Module):
    """
    Fault Classification head.
    Outputs logits for num_classes.
    """
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout_p: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.relu(h)
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits


class AnomalyHead(nn.Module):
    """
    Anomaly Score head (Binary Classification / Continuous Score).
    Outputs a single logit.
    """
    def __init__(self, in_features: int, hidden_dim: int, dropout_p: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.relu(h)
        h = self.dropout(h)
        logit = self.fc2(h)
        return logit
