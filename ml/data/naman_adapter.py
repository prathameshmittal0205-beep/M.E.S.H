import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional

from ml.data.contract import CanonicalBatch


class NpzDataset(Dataset):
    """
    Dataset adapter for Naman's .npz format.
    Dynamically infers modalities from keys starting with 'modality_'.
    """
    def __init__(self, npz_path: str, native_modalities: List[str], dataset_name: str = "unknown"):
        self.npz_path = npz_path
        self.native_modalities = native_modalities
        self.dataset_name = dataset_name
        
        with np.load(npz_path, allow_pickle=False) as data:
            self.sample_ids = data["sample_ids"]
            self.n_samples = len(self.sample_ids)
            
            # Load modalities
            self.modalities = {}
            self.masks = {}
            for mod in self.native_modalities:
                mod_key = f"modality_{mod}"
                mask_key = f"mask_{mod}"
                if mod_key not in data:
                    raise KeyError(f"Expected modality '{mod_key}' missing from .npz")
                self.modalities[mod] = torch.from_numpy(data[mod_key])
                self.masks[mod] = torch.from_numpy(data[mask_key]).to(torch.float32)
                
            # Load targets if available
            self.target_rul = torch.from_numpy(data["target_rul"]) if "target_rul" in data else None
            self.target_degradation = torch.from_numpy(data["target_degradation"]) if "target_degradation" in data else None
            self.target_binary_failure = torch.from_numpy(data["target_binary_failure"]) if "target_binary_failure" in data else None
            self.target_fault_class = torch.from_numpy(data["target_fault_class"]) if "target_fault_class" in data else None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = {
            "sample_id": self.sample_ids[idx],
            "dataset_name": self.dataset_name,
            "modality_values": {mod: self.modalities[mod][idx] for mod in self.native_modalities},
            "modality_mask_list": [self.masks[mod][idx] for mod in self.native_modalities]
        }
        
        if self.target_rul is not None:
            sample["target_rul"] = self.target_rul[idx]
            
        if self.target_binary_failure is not None:
            sample["target_binary_failure"] = self.target_binary_failure[idx]
        if self.target_fault_class is not None:
            sample["target_fault_class"] = self.target_fault_class[idx]
            
        return sample


def canonical_collate_fn(batch: List[Dict[str, Any]], native_modalities: List[str]) -> CanonicalBatch:
    """Collates a list of dataset samples into a CanonicalBatch."""
    batch_size = len(batch)
    
    sample_ids = [s["sample_id"] for s in batch]
    dataset_name = batch[0]["dataset_name"]
    
    # Modality values
    modality_values = {}
    for mod in native_modalities:
        modality_values[mod] = torch.stack([s["modality_values"][mod] for s in batch], dim=0)
        
    # Modality mask: shape [B, num_modalities]
    modality_mask = torch.stack([torch.stack(s["modality_mask_list"]) for s in batch], dim=0)
    
    # Targets
    target_rul = None
    if "target_rul" in batch[0]:
        target_rul = torch.stack([s["target_rul"] for s in batch]).view(-1, 1)
        
    target_fault_class = None
    if "target_fault_class" in batch[0]:
        target_fault_class = torch.stack([s["target_fault_class"].clone().detach().to(torch.long) for s in batch])
        
    target_degradation = None
    if "target_binary_failure" in batch[0]:
        # Map AI4I's target_binary_failure to target_degradation for AnomalyHead (BCE)
        target_degradation = torch.stack([s["target_binary_failure"].clone().detach().to(torch.float32) for s in batch]).view(-1, 1)
        
    return CanonicalBatch(
        sample_id=sample_ids,
        source_dataset=[dataset_name] * batch_size,
        dataset_version=["1.0.0"] * batch_size,
        run_id=["unknown"] * batch_size,
        window_start=[0.0] * batch_size,
        window_end=[1.0] * batch_size,
        sampling_interval_seconds=[1.0] * batch_size,
        native_modalities=native_modalities,
        preprocessor_version=["1.0.0"] * batch_size,
        scaler_version=["1.0.0"] * batch_size,
        modality_values=modality_values,
        modality_mask=modality_mask,
        target_rul=target_rul,
        target_fault_class=target_fault_class,
        target_degradation=target_degradation
    )
