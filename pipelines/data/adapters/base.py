"""Base abstract dataset adapter for MESH.

Defines the contract for dataset-specific ingestion, physical channel mapping,
and parsing into structured sequences adhering to pipelines.data.schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np

from pipelines.data.schema import CanonicalSample, ModalityMetadata, Targets, WindowMetadata


class BaseDatasetAdapter(ABC):
    """Abstract interface for all MESH dataset adapters."""

    def __init__(self, dataset_id: str, dataset_version: str = "1.0.0"):
        self.dataset_id = dataset_id
        self.dataset_version = dataset_version

    @property
    @abstractmethod
    def supported_modalities(self) -> Dict[str, ModalityMetadata]:
        """Returns the dictionary of native modalities supported by this dataset."""
        pass

    @property
    @abstractmethod
    def supported_targets(self) -> List[str]:
        """Returns the list of valid target variables supported by this dataset."""
        pass

    @abstractmethod
    def load_raw(self, raw_path: Path, **kwargs: Any) -> pd.DataFrame:
        """Loads and performs initial structural ingestion of raw dataset files."""
        pass

    @abstractmethod
    def parse_runs(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> Dict[str, pd.DataFrame]:
        """Groups data by physical asset/run identifier, sorted chronologically."""
        pass

    @abstractmethod
    def extract_windows(
        self,
        run_data: Dict[str, pd.DataFrame],
        window_size: int = 20,
        stride: int = 5,
        preprocessor_version: str = "raw_unscaled",
        scaler_version: str = "none",
    ) -> List[CanonicalSample]:
        """Converts runs into canonical sliding-window samples of length window_size with stride."""
        pass
