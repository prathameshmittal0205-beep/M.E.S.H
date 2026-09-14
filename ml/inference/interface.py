from typing import Dict, Any, List, Optional
import torch
import numpy as np
from dataclasses import dataclass, field
import datetime

from ml.data.contract import CanonicalBatch
from ml.models.mesh_model import MESHModel
from ml.models.ai4i_model import AI4IModel

@dataclass
class PredictionBundle:
    """
    Standardized payload delivered to Sarthak's backend for downstream consumption.
    All scalar values are de-normalized (real units).
    """
    # System Metadata (Artifact Traceability)
    model_version: str
    checkpoint_id: str
    training_config_ref: str
    
    # Regression (Real Units)
    rul_mean_cycles: List[float]
    rul_variance_cycles: List[float]
    
    # Classification & Anomaly (Optional, depending on dataset/model)
    fault_probabilities: Optional[List[List[float]]] = None
    anomaly_probabilities: Optional[List[float]] = None
    
    # Explainability (Optional)
    attention_weights: Optional[List[Any]] = None
    prediction_timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

class MESHInferenceEngine:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        # Load Checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model_config = checkpoint['model_config']
        self.train_config = checkpoint['train_config']
        self.norm_stats = checkpoint['target_normalization']
        self.model_version = checkpoint.get('model_version', 'unknown')
        self.model_type = self.model_config.get('model_type', 'MESHModel')
        
        self.native_modalities = self.model_config['native_modalities']
        
        # Reconstruct Model strictly from checkpoint config
        if self.model_type == 'MESHModel':
            self.model = MESHModel(
                native_modalities=self.native_modalities,
                cnn_in_channels_map=self.model_config['modality_channels'],
                encoder_config=self.model_config['encoder_config'],
                fusion_config=self.model_config['fusion_config'],
                temporal_config=self.model_config['temporal_config'],
                heads_config=self.model_config['heads_config'],
                dropout_p=0.0 # Inference mode strictly disables dropout
            )
        elif self.model_type == 'AI4IModel':
            self.model = AI4IModel(
                native_modalities=self.native_modalities,
                modality_channels=self.model_config['modality_channels'],
                embed_dim=self.model_config['embed_dim'],
                num_heads=self.model_config['num_heads'],
                num_fault_classes=self.model_config['num_fault_classes'],
                dropout_p=0.0
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
            
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def predict(self, canonical_batch: CanonicalBatch, return_attention: bool = False) -> PredictionBundle:
        """
        Executes an inference pass on a canonical batch.
        Returns a PredictionBundle with real-unit values and traceability metadata.
        """
        modality_values = {k: v.to(self.device) for k, v in canonical_batch.modality_values.items()}
        modality_mask = canonical_batch.modality_mask.to(self.device)
        
        with torch.no_grad():
            preds, attn_weights = self.model(modality_values, modality_mask)
            
        # De-normalize RUL to real units
        rul_mean_stat = self.norm_stats['rul_mean']
        rul_std_stat = self.norm_stats['rul_std']
        
        rul_mean_real = (preds['rul_mean'] * rul_std_stat) + rul_mean_stat
        rul_var_real = preds['rul_variance'] * (rul_std_stat ** 2)
        
        # Process Classifications (if heads are active)
        fault_probs = None
        if 'fault_logits' in preds:
            fault_probs = torch.softmax(preds['fault_logits'], dim=-1).cpu().numpy().tolist()
            
        anomaly_probs = None
        if 'anomaly_logit' in preds:
            anomaly_probs = torch.sigmoid(preds['anomaly_logit']).cpu().numpy().tolist()
        
        # If attention is requested, zero out the missing modalities for clean consumption
        clean_attn = None
        if return_attention and attn_weights is not None:
            # Mask dimensions depend on static vs temporal fusion
            # We skip detailed zeroing here if it's too complex across models,
            # or just return it raw. The fusion layer already masked queries/keys.
            clean_attn = attn_weights.cpu().numpy().tolist()

        return PredictionBundle(
            model_version=self.model_version,
            checkpoint_id=self.checkpoint_path,
            training_config_ref=self.train_config.get("experiment_name", "unknown_training"),
            rul_mean_cycles=rul_mean_real.cpu().numpy().tolist(),
            rul_variance_cycles=rul_var_real.cpu().numpy().tolist(),
            fault_probabilities=fault_probs,
            anomaly_probabilities=anomaly_probs,
            attention_weights=clean_attn
        )
