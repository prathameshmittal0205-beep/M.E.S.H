import os
import json
import torch
import yaml
from datetime import datetime
import itertools
from torch.utils.data import DataLoader

from ml.models.mesh_model import MESHModel
from ml.models.ai4i_model import AI4IModel
from ml.data.naman_adapter import NpzDataset, canonical_collate_fn


def generate_dropout_conditions(native_modalities):
    conditions = {"0_dropped_baseline": []}
    for i, mod in enumerate(native_modalities):
        conditions[f"1_dropped_{mod}"] = [i]
    
    # 2 dropped
    for comb in itertools.combinations(range(len(native_modalities)), 2):
        name = f"2_dropped_{native_modalities[comb[0]]}_{native_modalities[comb[1]]}"
        conditions[name] = list(comb)
        
    conditions[f"{len(native_modalities)}_dropped_total_dropout"] = list(range(len(native_modalities)))
    return conditions


def run_cmapss_experiment():
    print("\n=== CMAPSS Missing-Modality Experiment (UNTRAINED) ===")
    device = torch.device("cpu")
    
    with open("configs/model/cmapss_model.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    native_modalities = config['native_modalities']
    
    model = MESHModel(
        native_modalities=native_modalities,
        cnn_in_channels_map=config['modality_channels'],
        encoder_config=config['encoder_config'],
        fusion_config=config['fusion_config'],
        temporal_config=config['temporal_config'],
        heads_config=config['heads_config'],
        dropout_p=0.0
    ).to(device)
    model.eval()
    
    ds = NpzDataset("tests/fixtures/dummy_cmapss.npz", native_modalities, "cmapss")
    loader = DataLoader(ds, batch_size=10, collate_fn=lambda b: canonical_collate_fn(b, native_modalities))
    batch = next(iter(loader))
    
    base_values = {k: v.to(device) for k, v in batch.modality_values.items()}
    conditions = generate_dropout_conditions(native_modalities)
    
    with torch.no_grad():
        for cond_name, dropped_indices in conditions.items():
            if not cond_name.startswith("0") and not cond_name.startswith(f"{len(native_modalities)}"):
                if "1_dropped" not in cond_name:
                    continue # keep output brief
                    
            cond_mask = torch.ones(batch.modality_mask.shape[0], len(native_modalities)).to(device)
            for idx in dropped_indices:
                cond_mask[:, idx] = 0.0
                
            preds, _ = model(base_values, cond_mask)
            rul_mean = preds['rul_mean'].mean().item()
            rul_var = preds['rul_variance'].mean().item()
            print(f"{cond_name:40s} | RUL Mean: {rul_mean:8.4f} | RUL Var: {rul_var:8.4f}")


def run_ai4i_experiment():
    print("\n=== AI4I Missing-Modality Experiment (UNTRAINED) ===")
    device = torch.device("cpu")
    
    with open("configs/model/ai4i_model.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    native_modalities = config['native_modalities']
    
    model = AI4IModel(
        native_modalities=native_modalities,
        modality_channels=config['modality_channels'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_fault_classes=config['num_fault_classes'],
        dropout_p=0.0
    ).to(device)
    model.eval()
    
    ds = NpzDataset("tests/fixtures/dummy_ai4i.npz", native_modalities, "ai4i")
    loader = DataLoader(ds, batch_size=10, collate_fn=lambda b: canonical_collate_fn(b, native_modalities))
    batch = next(iter(loader))
    
    base_values = {k: v.to(device) for k, v in batch.modality_values.items()}
    conditions = generate_dropout_conditions(native_modalities)
    
    with torch.no_grad():
        for cond_name, dropped_indices in conditions.items():
            if not cond_name.startswith("0") and not cond_name.startswith(f"{len(native_modalities)}"):
                if "1_dropped" not in cond_name:
                    continue
                    
            cond_mask = torch.ones(batch.modality_mask.shape[0], len(native_modalities)).to(device)
            for idx in dropped_indices:
                cond_mask[:, idx] = 0.0
                
            preds, _ = model(base_values, cond_mask)
            rul_mean = preds['rul_mean'].mean().item()
            rul_var = preds['rul_variance'].mean().item()
            print(f"{cond_name:40s} | RUL Mean: {rul_mean:8.4f} | RUL Var: {rul_var:8.4f}")

if __name__ == "__main__":
    run_cmapss_experiment()
    run_ai4i_experiment()
