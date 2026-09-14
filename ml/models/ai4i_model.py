import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import OrderedDict

from ml.models.fusion import MaskAwareCrossAttention
from ml.models.heads import RULHead, FaultClassificationHead, AnomalyHead


class StaticModalityEncoder(nn.Module):
    """
    Lightweight static encoder for non-temporal datasets like AI4I.
    Maps a static feature snapshot to a fixed embedding dimension.
    """
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, 1, in_channels]
        return self.net(x)


class AI4IModel(nn.Module):
    """
    Lightweight model architecture for static, non-temporal datasets (e.g. AI4I 2020).
    Reuses mask-aware fusion but skips CNNs, BiLSTMs, and temporal Transformers.
    """
    def __init__(
        self,
        native_modalities: List[str],
        modality_channels: Dict[str, int],
        embed_dim: int = 128,
        num_heads: int = 4,
        num_fault_classes: int = 6,
        enable_rul: bool = False,
        dropout_p: float = 0.15
    ):
        super().__init__()
        self.native_modalities = native_modalities
        self.enable_rul = enable_rul
        
        # 1. Per-modality static encoders
        self.encoders = nn.ModuleDict()
        for mod in native_modalities:
            if mod not in modality_channels:
                raise ValueError(f"Missing channel size for {mod}")
            self.encoders[mod] = StaticModalityEncoder(
                in_channels=modality_channels[mod],
                embed_dim=embed_dim
            )
            
        # 2. Mask-aware fusion (we keep seq_len=1 as a dummy axis)
        self.fusion = MaskAwareCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_p
        )
        
        # 3. Task Heads
        if self.enable_rul:
            self.rul_head = RULHead(in_features=embed_dim, hidden_dim=64, dropout_p=dropout_p)
        self.fault_head = FaultClassificationHead(in_features=embed_dim, num_classes=num_fault_classes, hidden_dim=64, dropout_p=dropout_p)
        self.anomaly_head = AnomalyHead(in_features=embed_dim, hidden_dim=64, dropout_p=dropout_p)

    def forward(self, modality_values: Dict[str, torch.Tensor], mask: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            modality_values: dict mapping modality name to tensor of shape [batch_size, 1, C_m]
            mask: tensor of shape [batch_size, num_modalities]
            
        Returns:
            outputs: dictionary of predictions
            attn_weights: attention weights from fusion
        """
        B = next(iter(modality_values.values())).shape[0]
        device = next(iter(modality_values.values())).device
        
        # 1. Encode each modality independently
        embeddings = []
        for mod in self.native_modalities:
            x = modality_values[mod] # [B, 1, C_m]
            emb = self.encoders[mod](x) # [B, 1, embed_dim]
            embeddings.append(emb)
            
        # Stack into [B, seq_len=1, num_modalities, embed_dim]
        # We need dim=2 because emb is [B, 1, embed_dim], so stacking on dim=2 gives [B, 1, num_modalities, embed_dim]
        stacked_embeddings = torch.stack(embeddings, dim=2)
        
        # 2. Fuse modalities
        fused_embedding, attn_weights = self.fusion(stacked_embeddings, mask)
        
        # Squeeze the dummy sequence dimension (seq_len=1)
        # fused_embedding shape is [B, 1, embed_dim]
        fused_embedding = fused_embedding.squeeze(1) # [B, embed_dim]
        
        # 3. Task Heads
        outputs = {}
        
        # RUL (Mean and Variance)
        if self.enable_rul:
            rul_mean, rul_var = self.rul_head(fused_embedding)
            outputs['rul_mean'] = rul_mean
            outputs['rul_variance'] = rul_var
        
        # Fault Class Logits
        fault_logits = self.fault_head(fused_embedding)
        outputs['fault_logits'] = fault_logits
        
        # Anomaly Logits
        anomaly_logit = self.anomaly_head(fused_embedding)
        outputs['anomaly_logit'] = anomaly_logit
        
        return outputs, attn_weights
