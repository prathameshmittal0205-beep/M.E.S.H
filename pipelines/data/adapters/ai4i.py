"""AI4I 2020 Predictive Maintenance Dataset Adapter for MESH.

Handles the synthetic benchmark dataset per docs/data/DATA_PIPELINE.md,
docs/data/DATASET_STRATEGY.md, and docs/NO_HALLUCINATION_POLICY.md.

Note: AI4I 2020 is explicitly documented as synthetic tabular multi-sensor data
(docs/MESH_DATASET_DECISION_MATRIX.md). Product IDs are 100% unique (1 observation per machine).
Sliding windows across unrelated rows are strictly forbidden to prevent fabricating
artificial temporal continuity across different assets. Each sample is parsed as a
single static observation (T=1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np

from pipelines.data.adapters.base import BaseDatasetAdapter
from pipelines.data.schema import (
    CanonicalSample,
    ModalityMetadata,
    Targets,
    WindowMetadata,
)

AI4I_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


class AI4IAdapter(BaseDatasetAdapter):
    """Adapter for AI4I 2020 Predictive Maintenance synthetic tabular benchmark."""

    def __init__(self, dataset_version: str = "1.0.0"):
        super().__init__(dataset_id="ai4i2020", dataset_version=dataset_version)

    @property
    def supported_modalities(self) -> Dict[str, ModalityMetadata]:
        """Returns native physical sensor modality groupings for AI4I."""
        return {
            "temperature": ModalityMetadata(
                name="temperature",
                channels=["Air temperature [K]", "Process temperature [K]"],
                units=["K", "K"],
                description="Ambient air and milling process temperatures",
            ),
            "speed": ModalityMetadata(
                name="speed",
                channels=["Rotational speed [rpm]"],
                units=["rpm"],
                description="Spindle rotational speed",
            ),
            "torque": ModalityMetadata(
                name="torque",
                channels=["Torque [Nm]"],
                units=["Nm"],
                description="Spindle mechanical torque",
            ),
            "tool_wear": ModalityMetadata(
                name="tool_wear",
                channels=["Tool wear [min]"],
                units=["min"],
                description="Cumulative tool wear duration in minutes",
            ),
        }

    @property
    def supported_targets(self) -> List[str]:
        """Valid targets supported by AI4I 2020. RUL is NOT supported."""
        return ["target_binary_failure", "target_fault_class", "target_degradation"]

    def load_raw(self, raw_path: Path, **kwargs: Any) -> pd.DataFrame:
        """Loads the raw AI4I 2020 CSV file."""
        if not raw_path.exists():
            raise FileNotFoundError(f"AI4I raw CSV not found: {raw_path}")

        df = pd.read_csv(raw_path)
        return df

    def parse_runs(
        self,
        df: pd.DataFrame,
        run_group_col: Optional[str] = "Type",
        **kwargs: Any,
    ) -> Dict[str, pd.DataFrame]:
        """Groups AI4I records by product variant type or preserves individual observations.

        Since every row has a unique Product ID, there is no physical multi-cycle run.
        Groups are partitioned by product variant Type (L, M, H) to preserve batch structures.
        """
        runs = {}
        if run_group_col and run_group_col in df.columns:
            for group_name, group_df in df.groupby(run_group_col):
                runs[f"type_{group_name}"] = group_df.sort_values("UDI").copy().reset_index(drop=True)
        else:
            runs["all_products"] = df.sort_values("UDI").copy().reset_index(drop=True)
        return runs

    def extract_windows(
        self,
        run_data: Dict[str, pd.DataFrame],
        window_size: int = 1,
        stride: int = 1,
        preprocessor_version: str = "raw_unscaled",
        scaler_version: str = "none",
    ) -> List[CanonicalSample]:
        """Extracts canonical samples from AI4I 2020.

        IMPORTANT ENGINEERING & INTEGRITY RULE:
        AI4I 2020 consists of 10,000 independent synthetic observations with 10,000 unique
        Product IDs (1 observation per machine). There is NO continuous temporal trajectory
        per asset. Sliding multi-timestep windows (e.g. window_size=20) across unrelated
        Product IDs would stitch together distinct machines and fabricate artificial temporal
        continuity in direct violation of docs/NO_HALLUCINATION_POLICY.md and docs/data/DATASET_STRATEGY.md.

        Therefore, AI4I samples are strictly extracted as single static observations (window_size=1).
        If window_size > 1 is requested, a ValueError is raised to prevent silent data corruption.
        """
        if window_size > 1:
            raise ValueError(
                f"Cannot extract window_size={window_size} on AI4I 2020. "
                "AI4I consists of 10,000 independent single-snapshot machines (10,000 unique Product IDs). "
                "Sliding windows across unrelated machines violates docs/NO_HALLUCINATION_POLICY.md. "
                "Use window_size=1 for static tabular observation extraction."
            )

        modalities = self.supported_modalities
        canonical_samples: List[CanonicalSample] = []

        for group_id, group_df in run_data.items():
            for idx in range(len(group_df)):
                row = group_df.iloc[idx]
                udi = int(row["UDI"])
                product_id = str(row["Product ID"])

                # Targets
                binary_failure = int(row["Machine failure"])
                degradation_val = float(row["Tool wear [min]"])

                # Determine fault class
                fault_class = 0  # Normal / healthy
                for f_idx, mode in enumerate(FAILURE_MODES, start=1):
                    if int(row[mode]) == 1:
                        fault_class = f_idx
                        break

                # Extract per-modality numpy arrays of shape (T=1, C_m)
                modality_values: Dict[str, np.ndarray] = {}
                modality_mask: Dict[str, int] = {}

                for mod_name, mod_meta in modalities.items():
                    # Shape is (1, C_m) representing a single static observation timestep
                    vals = row[mod_meta.channels].to_numpy(dtype=np.float32).reshape(1, -1)
                    modality_values[mod_name] = vals
                    modality_mask[mod_name] = 1  # Available

                # Static feature: Type encoded
                type_val = str(row.get("Type", "L"))
                type_code = 0.0 if type_val == "L" else (1.0 if type_val == "M" else 2.0)

                sample_id = f"{self.dataset_id}_{product_id}_u{udi:05d}"
                metadata = WindowMetadata(
                    dataset_id=self.dataset_id,
                    dataset_version=self.dataset_version,
                    run_id=product_id,
                    asset_id=product_id,
                    window_start=udi,
                    window_end=udi,
                    sampling_interval_seconds=None,
                    operational_condition=f"Type_{type_val}",
                )

                targets = Targets(
                    target_rul=None,  # Strictly None: no run-to-failure protocol exists
                    target_fault_class=fault_class,
                    target_binary_failure=binary_failure,
                    target_degradation=degradation_val,
                )

                sample = CanonicalSample(
                    sample_id=sample_id,
                    metadata=metadata,
                    modality_values=modality_values,
                    modality_mask=modality_mask,
                    targets=targets,
                    preprocessor_version=preprocessor_version,
                    scaler_version=scaler_version,
                    static_features={"type_code": type_code},
                )
                sample.validate(expected_window_length=1)
                canonical_samples.append(sample)

        return canonical_samples
