from typing import Dict, Any, List, Optional
import torch
import numpy as np
from dataclasses import dataclass, field
import datetime

from ml.data.contract import CanonicalBatch
from ml.models.mesh_model import MESHModel

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
    
    # Classification & Anomaly
    fault_probabilities: List[List[float]]
    anomaly_probabilities: List[float]
    
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
        
        self.native_modalities = ["temperature", "tool_wear", "rotational_speed", "torque"]
        self.cnn_in_channels_map = {m: 1 for m in self.native_modalities}
        
        # Reconstruct Model strictly from checkpoint config
        self.model = MESHModel(
            native_modalities=self.native_modalities,
            cnn_in_channels_map=self.cnn_in_channels_map,
            encoder_config=self.model_config['encoder'],
            fusion_config=self.model_config['fusion'],
            temporal_config=self.model_config['temporal'],
            heads_config=self.model_config['heads'],
            dropout_p=0.0 # Inference mode strictly disables dropout
        ).to(self.device)
        
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
        
        # Process Classifications
        fault_probs = torch.softmax(preds['fault_logits'], dim=-1)
        anomaly_probs = torch.sigmoid(preds['anomaly_logit'])
        
        # If attention is requested, zero out the missing modalities for clean consumption
        clean_attn = None
        if return_attention and attn_weights is not None:
            mask_expanded = modality_mask.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
            clean_attn = (attn_weights * mask_expanded).cpu().numpy().tolist()

        return PredictionBundle(
            model_version=self.model_version,
            checkpoint_id=self.checkpoint_path,
            training_config_ref=self.train_config.get("config_name", "da1_training"),
            rul_mean_cycles=rul_mean_real.cpu().numpy().tolist(),
            rul_variance_cycles=rul_var_real.cpu().numpy().tolist(),
            fault_probabilities=fault_probs.cpu().numpy().tolist(),
            anomaly_probabilities=anomaly_probs.cpu().numpy().tolist(),
            attention_weights=clean_attn
        )

if __name__ == "__main__":
    import json
    from ml.data.mock_canonical_batch import generate_mock_canonical_batch
    
    print("=== MESH INFERENCE INTERFACE TEST ===")
    
    engine = MESHInferenceEngine("checkpoints/model_best.pt")
    
    # Simulate an incoming backend request with 2 samples
    batch = generate_mock_canonical_batch(batch_size=2, window_size=20)
    
    print("Executing predict(return_attention=True)...")
    bundle = engine.predict(batch, return_attention=True)
    
    # We serialize the dataclass to dict for clean printing (simulating a JSON payload)
    from dataclasses import asdict
    bundle_dict = asdict(bundle)
    
    print("\n[PredictionBundle Payload Delivered to Sarthak's Backend]")
    print(json.dumps(bundle_dict, indent=2))
