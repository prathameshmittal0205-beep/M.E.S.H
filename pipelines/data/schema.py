"""Canonical data contracts, schemas, and container models for MESH.

Defines the exact structures agreed in docs/data/DATA_PIPELINE.md,
docs/team/INTEGRATION_CONTRACT.md, and docs/model/MODEL_SPEC.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class ModalityMetadata:
    """Metadata describing a physical modality channel grouping."""
    name: str
    channels: List[str]
    units: List[str]
    sampling_rate_hz: Optional[float] = None
    description: str = ""


@dataclass(frozen=True)
class ModalityMask:
    """Binary availability mask across modalities.
    
    1 = available, 0 = unavailable per docs/data/DATA_PIPELINE.md.
    """
    flags: Dict[str, int]

    def is_available(self, modality_name: str) -> bool:
        return self.flags.get(modality_name, 0) == 1

    def to_vector(self, modality_order: Sequence[str]) -> np.ndarray:
        return np.array([self.flags.get(m, 0) for m in modality_order], dtype=np.float32)

    def to_dict(self) -> Dict[str, int]:
        return dict(self.flags)


@dataclass(frozen=True)
class WindowMetadata:
    """Traceability metadata for a single temporal window."""
    dataset_id: str
    dataset_version: str
    run_id: str
    window_start: int
    window_end: int
    asset_id: Optional[str] = None
    sampling_interval_seconds: Optional[float] = None
    operational_condition: Optional[str] = None
    missingness_type: str = "none"  # "none" | "natural" | "synthetic_controlled_dropout"
    dropped_modalities: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Targets:
    """Dataset-supported target variables. Fields are nullable if not supported."""
    target_rul: Optional[float] = None
    target_fault_class: Optional[int] = None
    target_binary_failure: Optional[int] = None
    target_degradation: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[Union[float, int]]]:
        return {
            "target_rul": self.target_rul,
            "target_fault_class": self.target_fault_class,
            "target_binary_failure": self.target_binary_failure,
            "target_degradation": self.target_degradation,
        }


@dataclass
class CanonicalSample:
    """A single canonical windowed sample produced by Naman's data pipeline.

    Conforms directly to docs/data/DATA_PIPELINE.md (Canonical sample concept).
    """
    sample_id: str
    metadata: WindowMetadata
    # Dict mapping modality_name -> numpy array of shape (T, C_m)
    # Default contract: T = 20 timesteps
    modality_values: Dict[str, np.ndarray]
    # Dict mapping modality_name -> 1 (available) or 0 (unavailable)
    modality_mask: Dict[str, int]
    targets: Targets
    preprocessor_version: str
    scaler_version: str
    static_features: Optional[Dict[str, float]] = None

    def validate(self, expected_window_length: int = 20) -> None:
        """Validates internal shape and mask consistency."""
        for mod_name, array in self.modality_values.items():
            if array.ndim != 2:
                raise ValueError(
                    f"Modality '{mod_name}' in sample '{self.sample_id}' must be 2D (T, C_m), got ndim={array.ndim}"
                )
            if array.shape[0] != expected_window_length:
                raise ValueError(
                    f"Modality '{mod_name}' in sample '{self.sample_id}' has T={array.shape[0]}, expected {expected_window_length}"
                )
            if mod_name not in self.modality_mask:
                raise ValueError(
                    f"Modality '{mod_name}' missing from modality_mask in sample '{self.sample_id}'"
                )
            mask_val = self.modality_mask[mod_name]
            if mask_val not in (0, 1):
                raise ValueError(
                    f"Modality mask for '{mod_name}' must be 0 or 1, got {mask_val}"
                )


@dataclass
class ModalityBatch:
    """Collated batch of canonical samples ready for model consumption (PyTorch / NumPy).

    Used directly by Prathamesh's ML pipeline and Sarthak's FastAPI inference endpoint.
    """
    sample_ids: List[str]
    # Dict mapping modality_name -> tensor/array of shape (Batch_Size, T=20, C_m)
    modality_tensors: Dict[str, np.ndarray]
    # Binary mask matrix of shape (Batch_Size, Num_Modalities) or Dict[str, np.ndarray of shape (Batch_Size,)]
    modality_masks: Dict[str, np.ndarray]
    # Targets dictionary with batched arrays
    targets: Dict[str, np.ndarray]
    # Traceability metadata per sample in batch
    metadata: List[WindowMetadata]


@dataclass(frozen=True)
class SplitManifest:
    """Immutable split manifest recording asset/run-level train/val/test allocation."""
    dataset_id: str
    dataset_version: str
    split_strategy: str
    train_runs: List[str]
    val_runs: List[str]
    test_runs: List[str]
    leakage_verified: bool
    created_at: str
    notes: str = ""

    def verify_no_overlap(self) -> bool:
        """Verifies that train, val, and test asset/run sets are strictly disjoint."""
        s_train = set(self.train_runs)
        s_val = set(self.val_runs)
        s_test = set(self.test_runs)
        train_val = s_train.intersection(s_val)
        train_test = s_train.intersection(s_test)
        val_test = s_val.intersection(s_test)
        if train_val or train_test or val_test:
            raise ValueError(
                f"Split leakage detected! Overlaps: train-val={train_val}, train-test={train_test}, val-test={val_test}"
            )
        return True


@dataclass(frozen=True)
class DataQualityReport:
    """Data quality validation summary conforming to docs/data/DATA_PIPELINE.md."""
    dataset_id: str
    total_records: int
    total_runs: int
    missing_value_count: int
    duplicate_records_count: int
    monotonicity_violations: int
    invalid_sensor_readings: int
    sensor_availability_summary: Dict[str, float]
    passed_qa: bool
    notes: List[str] = field(default_factory=list)
