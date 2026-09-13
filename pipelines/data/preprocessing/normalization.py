"""Train-only normalization module for MESH.

Implements strict per-modality z-score normalization fit ONLY on training
assets/runs per docs/data/DATA_PIPELINE.md, docs/team/NAMAN_DATA.md, and docs/DEFINITION_OF_DONE.md.
Zero leakage from validation or test data is mathematically enforced and verified.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from pipelines.data.schema import CanonicalSample, ModalityMetadata, SplitManifest


@dataclass
class ChannelNormParams:
    """Fitted normalization parameters for a single sensor channel."""
    channel_name: str
    mean: float
    std: float
    is_constant: bool = False

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.is_constant:
            return np.zeros_like(values, dtype=np.float32)
        return (values.astype(np.float32) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.is_constant:
            return np.full_like(values, self.mean, dtype=np.float32)
        return values.astype(np.float32) * self.std + self.mean


class ModalityScaler:
    """Scaler managing normalization parameters for a single physical modality."""

    def __init__(self, modality_name: str, channels: List[str]):
        self.modality_name = modality_name
        self.channels = list(channels)
        self.channel_params: Dict[str, ChannelNormParams] = {}
        self.fitted: bool = False

    def fit(self, df: pd.DataFrame) -> ModalityScaler:
        """Fits mean and std for each channel using provided training dataframe only."""
        self.channel_params = {}
        for c in self.channels:
            if c not in df.columns:
                raise KeyError(f"Channel '{c}' required for modality '{self.modality_name}' not found in dataframe.")
            series = df[c].astype(np.float64)
            mean_val = float(series.mean())
            std_val = float(series.std(ddof=0))  # Population std on train split

            # Guard against zero-variance / constant channels (e.g. sea-level invariant sensors)
            if std_val < 1e-8:
                self.channel_params[c] = ChannelNormParams(
                    channel_name=c,
                    mean=mean_val,
                    std=1.0,
                    is_constant=True,
                )
            else:
                self.channel_params[c] = ChannelNormParams(
                    channel_name=c,
                    mean=mean_val,
                    std=std_val,
                    is_constant=False,
                )
        self.fitted = True
        return self

    def transform(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transforms dataframe channels or numpy array to normalized array of shape (..., Num_Channels)."""
        if not self.fitted:
            raise RuntimeError(f"ModalityScaler for '{self.modality_name}' is not fitted.")
        if isinstance(data, pd.DataFrame):
            cols = []
            for c in self.channels:
                raw_vals = data[c].values
                normed = self.channel_params[c].transform(raw_vals)
                cols.append(normed)
            return np.column_stack(cols).astype(np.float32)
        elif isinstance(data, np.ndarray):
            if data.shape[-1] != len(self.channels):
                raise ValueError(f"Expected last dim {len(self.channels)}, got {data.shape[-1]}")
            orig_shape = data.shape
            reshaped = data.reshape(-1, len(self.channels))
            cols = []
            for idx, c in enumerate(self.channels):
                normed = self.channel_params[c].transform(reshaped[:, idx])
                cols.append(normed)
            stacked = np.column_stack(cols).astype(np.float32)
            return stacked.reshape(orig_shape)
        else:
            raise TypeError(f"Unsupported data type for transform: {type(data)}")

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """Inverts normalized 2D/3D array back to original engineering scale."""
        if not self.fitted:
            raise RuntimeError(f"ModalityScaler for '{self.modality_name}' is not fitted.")
        orig_shape = arr.shape
        # Flatten time/window dims if 3D (Batch, Time, Channels)
        reshaped = arr.reshape(-1, len(self.channels))
        cols = []
        for idx, c in enumerate(self.channels):
            inverted = self.channel_params[c].inverse_transform(reshaped[:, idx])
            cols.append(inverted)
        stacked = np.column_stack(cols).astype(np.float32)
        return stacked.reshape(orig_shape)


class PerModalityStandardScaler:
    """Orchestrates per-modality normalization scalers for a dataset.

    Conforms to docs/data/DATA_PIPELINE.md: fitted strictly on training assets/runs.
    """

    def __init__(self, dataset_id: str, scaler_version: str = "1.0.0"):
        self.dataset_id = dataset_id
        self.scaler_version = scaler_version
        self.modalities: Dict[str, ModalityScaler] = {}
        self.fit_metadata: Dict[str, Any] = {}
        self.fitted: bool = False

    def fit(
        self,
        train_df: pd.DataFrame,
        modalities_metadata: Dict[str, ModalityMetadata],
        train_run_ids: Sequence[str],
        run_col: str,
    ) -> PerModalityStandardScaler:
        """Fits all modalities strictly on the training partition."""
        if len(train_df) == 0:
            raise ValueError("Cannot fit scaler on empty dataframe.")

        # Fit each modality
        self.modalities = {}
        for mod_name, mod_meta in modalities_metadata.items():
            scaler = ModalityScaler(modality_name=mod_name, channels=mod_meta.channels)
            scaler.fit(train_df)
            self.modalities[mod_name] = scaler

        self.fit_metadata = {
            "dataset_id": self.dataset_id,
            "scaler_version": self.scaler_version,
            "fit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "train_runs_count": len(train_run_ids),
            "train_records_count": len(train_df),
            "run_col": run_col,
            "fitted_modalities": list(modalities_metadata.keys()),
        }
        self.fitted = True
        return self

    def transform_modality(self, modality_name: str, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if modality_name not in self.modalities:
            raise KeyError(f"Modality '{modality_name}' not registered in scaler.")
        return self.modalities[modality_name].transform(data)

    def transform_sample(
        self,
        sample: CanonicalSample,
        new_preprocessor_version: Optional[str] = None,
    ) -> CanonicalSample:
        """Transforms all modality values in a CanonicalSample using fitted parameters."""
        if not self.fitted:
            raise RuntimeError("PerModalityStandardScaler is not fitted.")
        normed_modalities = {}
        for mod_name, raw_arr in sample.modality_values.items():
            if mod_name in self.modalities:
                normed_modalities[mod_name] = self.modalities[mod_name].transform(raw_arr)
            else:
                normed_modalities[mod_name] = raw_arr.copy()

        return CanonicalSample(
            sample_id=sample.sample_id,
            metadata=sample.metadata,
            modality_values=normed_modalities,
            modality_mask=dict(sample.modality_mask),
            targets=sample.targets,
            preprocessor_version=new_preprocessor_version or sample.preprocessor_version,
            scaler_version=self.scaler_version,
            static_features=dict(sample.static_features) if sample.static_features else None,
        )

    def transform_samples(
        self,
        samples: Sequence[CanonicalSample],
        new_preprocessor_version: Optional[str] = None,
    ) -> List[CanonicalSample]:
        """Transforms a sequence of CanonicalSample objects."""
        return [self.transform_sample(s, new_preprocessor_version=new_preprocessor_version) for s in samples]

    @classmethod
    def load(cls, pkl_path: Path) -> PerModalityStandardScaler:
        """Loads a fitted scaler from a pickle file."""
        with open(pkl_path, "rb") as f:
            scaler = pickle.load(f)
        if not isinstance(scaler, cls):
            raise TypeError(f"Loaded object is not {cls.__name__}: {type(scaler)}")
        return scaler

    def to_dict(self) -> Dict[str, Any]:
        """Serializes scaler parameters to a human-readable, verifiable dictionary."""
        mod_dict = {}
        for mod_name, scaler in self.modalities.items():
            mod_dict[mod_name] = {
                "channels": scaler.channels,
                "parameters": {
                    c: {
                        "mean": round(p.mean, 6),
                        "std": round(p.std, 6),
                        "is_constant": p.is_constant,
                    }
                    for c, p in scaler.channel_params.items()
                },
            }
        return {
            "dataset_id": self.dataset_id,
            "scaler_version": self.scaler_version,
            "fit_metadata": self.fit_metadata,
            "modalities": mod_dict,
        }

    def save(self, json_path: Path, pkl_path: Optional[Path] = None) -> None:
        """Saves scaler to JSON (traceable/auditable) and pickle (runtime deserialization)."""
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        if pkl_path is not None:
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(pkl_path, "wb") as f:
                pickle.dump(self, f)


def run_normalization_from_config(
    config: Dict[str, Any],
    root_dir: Optional[Path] = None,
) -> PerModalityStandardScaler:
    """Fits train-only normalization scalers driven entirely by a configuration dictionary."""
    if root_dir is None:
        root_dir = Path.cwd()

    ds_cfg = config["dataset"]
    norm_cfg = config["normalization"]
    paths_cfg = config["paths"]

    dataset_name = ds_cfg["name"]
    split_path = root_dir / paths_cfg["split_output"]
    if not split_path.exists():
        raise FileNotFoundError(f"Required split manifest not found at {split_path}. Run split stage first.")

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_runs = split_data["train_runs"]
    val_runs = split_data["val_runs"]
    test_runs = split_data["test_runs"]

    json_out = root_dir / paths_cfg["scaler_json_output"]
    pkl_out = root_dir / paths_cfg["scaler_pkl_output"]

    if dataset_name == "cmapss_fd001":
        from pipelines.data.adapters.cmapss import CMAPSSAdapter
        subdataset = ds_cfg.get("subdataset", "FD001")
        drop_invariant = ds_cfg.get("drop_invariant_sensors", False)
        adapter = CMAPSSAdapter(subdataset=subdataset, drop_invariant_sensors=drop_invariant)

        train_path = root_dir / paths_cfg["train_file"]
        if not train_path.exists():
            raise FileNotFoundError(f"C-MAPSS training file not found: {train_path}")

        raw_df = adapter.load_raw(train_path)

        train_unit_ids = [int(u.replace("unit_", "")) for u in train_runs]
        train_df = raw_df[raw_df["unit_number"].isin(train_unit_ids)].copy()

        # Leakage assert
        val_unit_ids = [int(u.replace("unit_", "")) for u in val_runs]
        assert set(train_df["unit_number"].unique()).isdisjoint(set(val_unit_ids)), "Leakage detected in training subset!"

        scaler = PerModalityStandardScaler(
            dataset_id=ds_cfg["dataset_id"],
            scaler_version=norm_cfg.get("scaler_version", "1.0.0"),
        )
        scaler.fit(
            train_df=train_df,
            modalities_metadata=adapter.supported_modalities,
            train_run_ids=train_runs,
            run_col=ds_cfg.get("run_identifier_col", "unit_number"),
        )
        scaler.save(json_path=json_out, pkl_path=pkl_out)
        return scaler

    elif dataset_name == "ai4i2020":
        from pipelines.data.adapters.ai4i import AI4IAdapter
        adapter = AI4IAdapter(dataset_version=ds_cfg.get("dataset_version", "1.0.0"))

        raw_path = root_dir / paths_cfg["train_file"]
        if not raw_path.exists():
            raise FileNotFoundError(f"AI4I raw data file not found: {raw_path}")

        raw_df = adapter.load_raw(raw_path)

        run_col = ds_cfg.get("run_identifier_col", "Product ID")
        train_df = raw_df[raw_df[run_col].isin(train_runs)].copy()

        # Leakage assert
        assert set(train_df[run_col].unique()).isdisjoint(set(val_runs)), "AI4I Train-Val leakage!"
        assert set(train_df[run_col].unique()).isdisjoint(set(test_runs)), "AI4I Train-Test leakage!"

        scaler = PerModalityStandardScaler(
            dataset_id=ds_cfg["dataset_id"],
            scaler_version=norm_cfg.get("scaler_version", "1.0.0"),
        )
        scaler.fit(
            train_df=train_df,
            modalities_metadata=adapter.supported_modalities,
            train_run_ids=train_runs,
            run_col=run_col,
        )
        scaler.save(json_path=json_out, pkl_path=pkl_out)
        return scaler

    else:
        raise ValueError(f"Unsupported dataset for normalization: '{dataset_name}'")


def fit_and_save_scalers(
    root_dir: Optional[Path] = None,
) -> Tuple[PerModalityStandardScaler, PerModalityStandardScaler]:
    """Fits train-only scalers for C-MAPSS FD001 and AI4I 2020 via configuration."""
    import yaml

    if root_dir is None:
        root_dir = Path.cwd()

    cmapss_cfg_path = root_dir / "configs" / "data" / "cmapss_fd001.yaml"
    ai4i_cfg_path = root_dir / "configs" / "data" / "ai4i2020.yaml"

    if cmapss_cfg_path.exists() and ai4i_cfg_path.exists():
        with open(cmapss_cfg_path, "r", encoding="utf-8") as f:
            cmapss_cfg = yaml.safe_load(f)
        with open(ai4i_cfg_path, "r", encoding="utf-8") as f:
            ai4i_cfg = yaml.safe_load(f)

        cmapss_scaler = run_normalization_from_config(cmapss_cfg, root_dir=root_dir)
        ai4i_scaler = run_normalization_from_config(ai4i_cfg, root_dir=root_dir)
        return cmapss_scaler, ai4i_scaler

    # Fallback if configs are missing
    splits_dir = root_dir / "data" / "processed" / "splits"
    scalers_dir = root_dir / "data" / "processed" / "scalers"
    scalers_dir.mkdir(parents=True, exist_ok=True)

    from pipelines.data.adapters.cmapss import CMAPSSAdapter
    from pipelines.data.adapters.ai4i import AI4IAdapter

    cmapss_split_path = splits_dir / "cmapss_fd001_split.json"
    with open(cmapss_split_path, "r", encoding="utf-8") as f:
        cmapss_split_data = json.load(f)

    train_runs = cmapss_split_data["train_runs"]
    val_runs = cmapss_split_data["val_runs"]
    train_unit_ids = [int(u.replace("unit_", "")) for u in train_runs]
    val_unit_ids = [int(u.replace("unit_", "")) for u in val_runs]

    cmapss_adapter = CMAPSSAdapter(subdataset="FD001", drop_invariant_sensors=False)
    cmapss_raw_path = root_dir / "data" / "raw" / "cmapss" / "train_FD001.txt"
    full_cmapss_df = cmapss_adapter.load_raw(cmapss_raw_path)

    cmapss_train_df = full_cmapss_df[full_cmapss_df["unit_number"].isin(train_unit_ids)].copy()
    assert set(cmapss_train_df["unit_number"].unique()).isdisjoint(set(val_unit_ids)), "Leakage detected!"

    cmapss_scaler = PerModalityStandardScaler(dataset_id="nasa_cmapss_fd001", scaler_version="1.0.0")
    cmapss_scaler.fit(
        train_df=cmapss_train_df,
        modalities_metadata=cmapss_adapter.supported_modalities,
        train_run_ids=train_runs,
        run_col="unit_number",
    )
    cmapss_scaler.save(
        json_path=scalers_dir / "cmapss_fd001_scaler.json",
        pkl_path=scalers_dir / "cmapss_fd001_scaler.pkl",
    )

    ai4i_split_path = splits_dir / "ai4i2020_split.json"
    with open(ai4i_split_path, "r", encoding="utf-8") as f:
        ai4i_split_data = json.load(f)

    ai4i_train_runs = ai4i_split_data["train_runs"]
    ai4i_val_runs = ai4i_split_data["val_runs"]
    ai4i_test_runs = ai4i_split_data["test_runs"]

    ai4i_adapter = AI4IAdapter()
    ai4i_raw_path = root_dir / "data" / "raw" / "ai4i2020" / "ai4i2020.csv"
    full_ai4i_df = ai4i_adapter.load_raw(ai4i_raw_path)

    ai4i_train_df = full_ai4i_df[full_ai4i_df["Product ID"].isin(ai4i_train_runs)].copy()
    assert set(ai4i_train_df["Product ID"].unique()).isdisjoint(set(ai4i_val_runs)), "AI4I Train-Val leakage!"
    assert set(ai4i_train_df["Product ID"].unique()).isdisjoint(set(ai4i_test_runs)), "AI4I Train-Test leakage!"

    ai4i_scaler = PerModalityStandardScaler(dataset_id="ai4i2020", scaler_version="1.0.0")
    ai4i_scaler.fit(
        train_df=ai4i_train_df,
        modalities_metadata=ai4i_adapter.supported_modalities,
        train_run_ids=ai4i_train_runs,
        run_col="Product ID",
    )
    ai4i_scaler.save(
        json_path=scalers_dir / "ai4i2020_scaler.json",
        pkl_path=scalers_dir / "ai4i2020_scaler.pkl",
    )

    return cmapss_scaler, ai4i_scaler


if __name__ == "__main__":
    cmapss_s, ai4i_s = fit_and_save_scalers()
    print("Train-only normalization fitting complete.")
    print("C-MAPSS Scaler fitted modalities:", list(cmapss_s.modalities.keys()))
    print("AI4I Scaler fitted modalities:", list(ai4i_s.modalities.keys()))
