"""Sliding window extraction, normalization application, and split routing module for MESH.

Conforms to docs/data/DATA_PIPELINE.md and docs/team/NAMAN_DATA.md:
- Windowing configuration: T=20 timesteps, stride=5 (C-MAPSS) / T=1 static snapshot (AI4I).
- Normalization: applies train-only fitted scaler parameters to all splits (never refits).
- Split separation: strictly routes samples into isolated split subdirectories (train/val/test).
- Final canonical storage: collates samples into batched NumPy .npz arrays (ModalityBatch layout)
  alongside manifest.json, preserving samples.pkl as an intermediate artifact.
- Constant channels: verifies invariant sensors normalize to 0.0 without NaN or Inf.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from pipelines.data.schema import CanonicalSample, ModalityBatch, SplitManifest, WindowMetadata
from pipelines.data.adapters.cmapss import CMAPSSAdapter
from pipelines.data.adapters.ai4i import AI4IAdapter
from pipelines.data.preprocessing.normalization import PerModalityStandardScaler


def verify_normalized_sample(sample: CanonicalSample) -> None:
    """Verifies that normalized sample has no NaNs/Infs."""
    for mod_name, arr in sample.modality_values.items():
        if np.isnan(arr).any():
            raise ValueError(f"NaN detected in modality '{mod_name}' for sample '{sample.sample_id}'")
        if np.isinf(arr).any():
            raise ValueError(f"Inf detected in modality '{mod_name}' for sample '{sample.sample_id}'")


def save_canonical_split(
    samples: Sequence[CanonicalSample],
    split_dir: Path,
    dataset_id: str,
    split_name: str,
    window_size: int,
    stride: Optional[int],
    preprocessor_version: str,
    scaler_version: str,
    run_ids: Sequence[str],
    modalities_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Collates and serializes canonical samples to NumPy .npz, JSON manifest, and intermediate pickle.

    The .npz artifact provides portable, tensor-ready arrays for Prathamesh's ML pipeline:
    - modality_{name}: (N, T, C_m) float32 array
    - mask_{name}: (N,) int32 array
    - target_{name}: (N,) float32 or int32 array
    - sample_ids: (N,) string array
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    n_samples = len(samples)

    npz_data: Dict[str, np.ndarray] = {}

    if n_samples > 0:
        sample_ids = [s.sample_id for s in samples]
        npz_data["sample_ids"] = np.array(sample_ids, dtype=str)

        first_sample = samples[0]
        mod_names = list(first_sample.modality_values.keys())

        # Collate modality tensors and binary availability masks
        for mod in mod_names:
            mod_stacked = np.stack([s.modality_values[mod] for s in samples], axis=0).astype(np.float32)
            npz_data[f"modality_{mod}"] = mod_stacked
            mask_stacked = np.array([s.modality_mask.get(mod, 0) for s in samples], dtype=np.int32)
            npz_data[f"mask_{mod}"] = mask_stacked

        # Collate supported targets
        if any(s.targets.target_rul is not None for s in samples):
            ruls = [s.targets.target_rul if s.targets.target_rul is not None else np.nan for s in samples]
            npz_data["target_rul"] = np.array(ruls, dtype=np.float32)

        if any(s.targets.target_binary_failure is not None for s in samples):
            failures = [s.targets.target_binary_failure if s.targets.target_binary_failure is not None else -1 for s in samples]
            npz_data["target_binary_failure"] = np.array(failures, dtype=np.int32)

        if any(s.targets.target_fault_class is not None for s in samples):
            fault_classes = [s.targets.target_fault_class if s.targets.target_fault_class is not None else -1 for s in samples]
            npz_data["target_fault_class"] = np.array(fault_classes, dtype=np.int32)

        if any(s.targets.target_degradation is not None for s in samples):
            degradations = [s.targets.target_degradation if s.targets.target_degradation is not None else np.nan for s in samples]
            npz_data["target_degradation"] = np.array(degradations, dtype=np.float32)

    # 1. Save portable compressed NumPy .npz (Primary Canonical Artifact)
    npz_path = split_dir / "data.npz"
    np.savez_compressed(npz_path, **npz_data)

    # 2. Save Python dataclass list (Secondary Intermediate Artifact)
    samples_pkl = split_dir / "samples.pkl"
    with open(samples_pkl, "wb") as f:
        pickle.dump(list(samples), f)

    # 3. Build comprehensive JSON manifest
    modality_info = {}
    if n_samples > 0:
        for mod, meta in modalities_metadata.items():
            modality_info[mod] = {
                "channels": meta.channels,
                "units": meta.units,
                "shape": list(npz_data[f"modality_{mod}"].shape),
            }

    target_stats: Dict[str, Any] = {}
    if "target_rul" in npz_data:
        r_arr = npz_data["target_rul"]
        target_stats["rul_min"] = float(np.nanmin(r_arr))
        target_stats["rul_max"] = float(np.nanmax(r_arr))
        target_stats["rul_mean"] = round(float(np.nanmean(r_arr)), 4)

    if "target_binary_failure" in npz_data:
        f_arr = npz_data["target_binary_failure"]
        pos = int((f_arr == 1).sum())
        target_stats["total_failures"] = pos
        target_stats["failure_rate"] = round(pos / n_samples, 6) if n_samples > 0 else 0.0

    if "target_fault_class" in npz_data:
        fc_arr = npz_data["target_fault_class"]
        unique_classes, counts = np.unique(fc_arr, return_counts=True)
        target_stats["fault_class_distribution"] = {int(k): int(v) for k, v in zip(unique_classes, counts)}

    manifest_meta = {
        "dataset_id": dataset_id,
        "split_name": split_name,
        "sample_count": n_samples,
        "run_count": len(run_ids),
        "runs": sorted(list(run_ids)),
        "window_size": window_size,
        "stride": stride,
        "preprocessor_version": preprocessor_version,
        "scaler_version": scaler_version,
        "files": {
            "canonical_data_npz": "data.npz",
            "intermediate_pickle": "samples.pkl",
        },
        "modalities": modality_info,
        "target_statistics": target_stats,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    manifest_json = split_dir / "manifest.json"
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest_meta, f, indent=2)

    return manifest_meta


def load_split_modality_batch(split_dir: Path) -> ModalityBatch:
    """Loads a split's data.npz directly into a ModalityBatch container."""
    npz_path = split_dir / "data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Canonical data array not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        sample_ids = [str(sid) for sid in data["sample_ids"]]
        modality_tensors: Dict[str, np.ndarray] = {}
        modality_masks: Dict[str, np.ndarray] = {}
        targets: Dict[str, np.ndarray] = {}

        for k in data.files:
            if k.startswith("modality_"):
                mod_name = k.replace("modality_", "")
                modality_tensors[mod_name] = data[k]
            elif k.startswith("mask_"):
                mod_name = k.replace("mask_", "")
                modality_masks[mod_name] = data[k]
            elif k.startswith("target_"):
                target_name = k
                targets[target_name] = data[k]

    # Load metadata from samples.pkl if present
    metadata: List[WindowMetadata] = []
    pkl_path = split_dir / "samples.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            samples = pickle.load(f)
            metadata = [s.metadata for s in samples]

    return ModalityBatch(
        sample_ids=sample_ids,
        modality_tensors=modality_tensors,
        modality_masks=modality_masks,
        targets=targets,
        metadata=metadata,
    )


def process_cmapss_windows(
    root_dir: Optional[Path] = None,
    subdataset: str = "FD001",
    window_size: int = 20,
    stride: int = 5,
) -> Dict[str, Any]:
    """Generates normalized, split-separated canonical window datasets for C-MAPSS."""
    if root_dir is None:
        root_dir = Path.cwd()

    adapter = CMAPSSAdapter(subdataset=subdataset)
    split_path = root_dir / "data" / "processed" / "splits" / f"cmapss_{subdataset.lower()}_split.json"
    scaler_path = root_dir / "data" / "processed" / "scalers" / f"cmapss_{subdataset.lower()}_scaler.pkl"
    raw_train_path = root_dir / "data" / "raw" / "cmapss" / f"train_{subdataset}.txt"
    raw_test_path = root_dir / "data" / "raw" / "cmapss" / f"test_{subdataset}.txt"
    raw_rul_path = root_dir / "data" / "raw" / "cmapss" / f"RUL_{subdataset}.txt"

    # Load split manifest & train-only fitted scaler
    with open(split_path, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)

    train_run_ids = set(split_manifest["train_runs"])
    val_run_ids = set(split_manifest["val_runs"])
    test_run_ids = set(split_manifest["test_runs"])

    scaler = PerModalityStandardScaler.load(scaler_path)

    # 1. Load and parse raw training and validation runs
    train_df = adapter.load_raw(raw_train_path)
    all_train_runs = adapter.parse_runs(train_df, is_train=True)

    train_run_dict = {k: v for k, v in all_train_runs.items() if k in train_run_ids}
    val_run_dict = {k: v for k, v in all_train_runs.items() if k in val_run_ids}

    # 2. Load and parse raw test runs with explicit ground-truth RUL file
    test_df = adapter.load_raw(raw_test_path)
    all_test_runs = adapter.parse_runs(test_df, is_train=False, rul_file_path=raw_rul_path)

    test_run_dict = {}
    for k, v in all_test_runs.items():
        test_key = f"test_{k}"
        if test_key in test_run_ids or k in test_run_ids:
            test_run_dict[test_key] = v

    split_groups = {
        "train": train_run_dict,
        "val": val_run_dict,
        "test": test_run_dict,
    }

    base_output_dir = root_dir / "data" / "processed" / f"cmapss_{subdataset.lower()}"
    results_summary: Dict[str, Any] = {}

    preprocessor_version = f"v1.0_window{window_size}_stride{stride}"
    scaler_version = scaler.scaler_version

    for split_name, run_dict in split_groups.items():
        # Slicing raw windows
        raw_samples = adapter.extract_windows(
            run_data=run_dict,
            window_size=window_size,
            stride=stride,
            preprocessor_version="raw_unscaled",
            scaler_version="none",
        )

        # Applying train-fit normalization
        normalized_samples = scaler.transform_samples(
            samples=raw_samples,
            new_preprocessor_version=preprocessor_version,
        )

        # Verify no NaN/Inf and constant channels are exactly 0.0
        for s in normalized_samples:
            verify_normalized_sample(s)
            if "temperatures" in s.modality_values:
                np.testing.assert_allclose(s.modality_values["temperatures"][:, 0], 0.0, atol=1e-6)
            if "pressures" in s.modality_values:
                np.testing.assert_allclose(s.modality_values["pressures"][:, 0], 0.0, atol=1e-6)

        split_dir = base_output_dir / split_name
        manifest_meta = save_canonical_split(
            samples=normalized_samples,
            split_dir=split_dir,
            dataset_id=adapter.dataset_id,
            split_name=split_name,
            window_size=window_size,
            stride=stride,
            preprocessor_version=preprocessor_version,
            scaler_version=scaler_version,
            run_ids=list(run_dict.keys()),
            modalities_metadata=adapter.supported_modalities,
        )
        results_summary[split_name] = manifest_meta

    return results_summary


def run_windowing_from_config(
    config: Dict[str, Any],
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Extracts windows, applies train-fit scaler, and saves canonical .npz / manifest per config."""
    if root_dir is None:
        root_dir = Path.cwd()

    ds_cfg = config["dataset"]
    win_cfg = config["windowing"]
    paths_cfg = config["paths"]

    dataset_name = ds_cfg["name"]
    window_size = win_cfg["window_size"]
    stride = win_cfg.get("stride")
    preprocessor_version = win_cfg.get("preprocessor_version", f"v1.0_w{window_size}")

    split_path = root_dir / paths_cfg["split_output"]
    scaler_path = root_dir / paths_cfg["scaler_pkl_output"]
    base_output_dir = root_dir / paths_cfg["processed_dir"]

    if not split_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_path}. Run split stage first.")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler pickle not found: {scaler_path}. Run normalization stage first.")

    with open(split_path, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)

    scaler = PerModalityStandardScaler.load(scaler_path)
    scaler_version = scaler.scaler_version

    results_summary: Dict[str, Any] = {}

    if dataset_name == "cmapss_fd001":
        subdataset = ds_cfg.get("subdataset", "FD001")
        drop_invariant = ds_cfg.get("drop_invariant_sensors", False)
        adapter = CMAPSSAdapter(subdataset=subdataset, drop_invariant_sensors=drop_invariant)

        raw_train_path = root_dir / paths_cfg["train_file"]
        raw_test_path = root_dir / paths_cfg["test_file"]
        raw_rul_path = root_dir / paths_cfg["rul_file"]

        if not raw_train_path.exists():
            raise FileNotFoundError(f"C-MAPSS train file missing: {raw_train_path}")
        if not raw_test_path.exists():
            raise FileNotFoundError(f"C-MAPSS test file missing: {raw_test_path}")
        if not raw_rul_path.exists():
            raise FileNotFoundError(f"C-MAPSS RUL file missing: {raw_rul_path}")

        train_run_ids = set(split_manifest["train_runs"])
        val_run_ids = set(split_manifest["val_runs"])
        test_run_ids = set(split_manifest["test_runs"])

        train_df = adapter.load_raw(raw_train_path)
        all_train_runs = adapter.parse_runs(train_df, is_train=True)

        train_run_dict = {k: v for k, v in all_train_runs.items() if k in train_run_ids}
        val_run_dict = {k: v for k, v in all_train_runs.items() if k in val_run_ids}

        test_df = adapter.load_raw(raw_test_path)
        all_test_runs = adapter.parse_runs(test_df, is_train=False, rul_file_path=raw_rul_path)

        test_run_dict = {}
        for k, v in all_test_runs.items():
            test_key = f"test_{k}"
            if test_key in test_run_ids or k in test_run_ids:
                test_run_dict[test_key] = v

        split_groups = {
            "train": train_run_dict,
            "val": val_run_dict,
            "test": test_run_dict,
        }

        for split_name, run_dict in split_groups.items():
            raw_samples = adapter.extract_windows(
                run_data=run_dict,
                window_size=window_size,
                stride=stride,
                preprocessor_version="raw_unscaled",
                scaler_version="none",
            )
            normalized_samples = scaler.transform_samples(
                samples=raw_samples,
                new_preprocessor_version=preprocessor_version,
            )

            for s in normalized_samples:
                verify_normalized_sample(s)
                if not drop_invariant:
                    if "temperatures" in s.modality_values:
                        np.testing.assert_allclose(s.modality_values["temperatures"][:, 0], 0.0, atol=1e-6)
                    if "pressures" in s.modality_values:
                        np.testing.assert_allclose(s.modality_values["pressures"][:, 0], 0.0, atol=1e-6)

            split_dir = base_output_dir / split_name
            manifest_meta = save_canonical_split(
                samples=normalized_samples,
                split_dir=split_dir,
                dataset_id=adapter.dataset_id,
                split_name=split_name,
                window_size=window_size,
                stride=stride,
                preprocessor_version=preprocessor_version,
                scaler_version=scaler_version,
                run_ids=list(run_dict.keys()),
                modalities_metadata=adapter.supported_modalities,
            )
            results_summary[split_name] = manifest_meta

        return results_summary

    elif dataset_name == "ai4i2020":
        adapter = AI4IAdapter(dataset_version=ds_cfg.get("dataset_version", "1.0.0"))
        raw_path = root_dir / paths_cfg["train_file"]
        if not raw_path.exists():
            raise FileNotFoundError(f"AI4I raw file missing: {raw_path}")

        train_ids = set(split_manifest["train_runs"])
        val_ids = set(split_manifest["val_runs"])
        test_ids = set(split_manifest["test_runs"])

        df = adapter.load_raw(raw_path)
        train_df = df[df["Product ID"].isin(train_ids)].copy().reset_index(drop=True)
        val_df = df[df["Product ID"].isin(val_ids)].copy().reset_index(drop=True)
        test_df = df[df["Product ID"].isin(test_ids)].copy().reset_index(drop=True)

        split_groups = {
            "train": (train_ids, adapter.parse_runs(train_df)),
            "val": (val_ids, adapter.parse_runs(val_df)),
            "test": (test_ids, adapter.parse_runs(test_df)),
        }

        for split_name, (asset_id_set, run_dict) in split_groups.items():
            raw_samples = adapter.extract_windows(
                run_data=run_dict,
                preprocessor_version="raw_unscaled",
                scaler_version="none",
            )
            normalized_samples = scaler.transform_samples(
                samples=raw_samples,
                new_preprocessor_version=preprocessor_version,
            )
            for s in normalized_samples:
                verify_normalized_sample(s)

            split_dir = base_output_dir / split_name
            manifest_meta = save_canonical_split(
                samples=normalized_samples,
                split_dir=split_dir,
                dataset_id=adapter.dataset_id,
                split_name=split_name,
                window_size=window_size,
                stride=stride,
                preprocessor_version=preprocessor_version,
                scaler_version=scaler_version,
                run_ids=list(asset_id_set),
                modalities_metadata=adapter.supported_modalities,
            )
            results_summary[split_name] = manifest_meta

        return results_summary

    else:
        raise ValueError(f"Unsupported dataset for windowing: '{dataset_name}'")


def process_ai4i_windows(
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generates normalized, split-separated canonical static snapshot datasets for AI4I 2020."""
    if root_dir is None:
        root_dir = Path.cwd()

    import yaml
    ai4i_cfg_path = root_dir / "configs" / "data" / "ai4i2020.yaml"
    if ai4i_cfg_path.exists():
        with open(ai4i_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return run_windowing_from_config(cfg, root_dir=root_dir)

    adapter = AI4IAdapter()
    split_path = root_dir / "data" / "processed" / "splits" / "ai4i2020_split.json"
    scaler_path = root_dir / "data" / "processed" / "scalers" / "ai4i2020_scaler.pkl"
    raw_path = root_dir / "data" / "raw" / "ai4i2020" / "ai4i2020.csv"

    with open(split_path, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)

    train_ids = set(split_manifest["train_runs"])
    val_ids = set(split_manifest["val_runs"])
    test_ids = set(split_manifest["test_runs"])

    scaler = PerModalityStandardScaler.load(scaler_path)
    df = adapter.load_raw(raw_path)

    train_df = df[df["Product ID"].isin(train_ids)].copy().reset_index(drop=True)
    val_df = df[df["Product ID"].isin(val_ids)].copy().reset_index(drop=True)
    test_df = df[df["Product ID"].isin(test_ids)].copy().reset_index(drop=True)

    split_groups = {
        "train": (train_ids, adapter.parse_runs(train_df)),
        "val": (val_ids, adapter.parse_runs(val_df)),
        "test": (test_ids, adapter.parse_runs(test_df)),
    }

    base_output_dir = root_dir / "data" / "processed" / "ai4i2020"
    results_summary: Dict[str, Any] = {}

    preprocessor_version = "v1.0_static_snapshot"
    scaler_version = scaler.scaler_version

    for split_name, (asset_id_set, run_dict) in split_groups.items():
        raw_samples = adapter.extract_windows(
            run_data=run_dict,
            preprocessor_version="raw_unscaled",
            scaler_version="none",
        )
        normalized_samples = scaler.transform_samples(
            samples=raw_samples,
            new_preprocessor_version=preprocessor_version,
        )
        for s in normalized_samples:
            verify_normalized_sample(s)

        split_dir = base_output_dir / split_name
        manifest_meta = save_canonical_split(
            samples=normalized_samples,
            split_dir=split_dir,
            dataset_id=adapter.dataset_id,
            split_name=split_name,
            window_size=1,
            stride=None,
            preprocessor_version=preprocessor_version,
            scaler_version=scaler_version,
            run_ids=list(asset_id_set),
            modalities_metadata=adapter.supported_modalities,
        )
        results_summary[split_name] = manifest_meta

    return results_summary


def run_stage7_pipeline(root_dir: Optional[Path] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Executes full Stage 7 pipeline for both datasets, outputting .npz and JSON manifests."""
    if root_dir is None:
        root_dir = Path.cwd()

    import yaml
    cmapss_cfg_path = root_dir / "configs" / "data" / "cmapss_fd001.yaml"
    ai4i_cfg_path = root_dir / "configs" / "data" / "ai4i2020.yaml"

    if cmapss_cfg_path.exists() and ai4i_cfg_path.exists():
        print("Executing Stage 7 via configs: Normalizing, collating .npz, and routing C-MAPSS FD001 windows...")
        with open(cmapss_cfg_path, "r", encoding="utf-8") as f:
            c_cfg = yaml.safe_load(f)
        cmapss_summary = run_windowing_from_config(c_cfg, root_dir=root_dir)

        print("Executing Stage 7 via configs: Normalizing, collating .npz, and routing AI4I 2020 static windows...")
        with open(ai4i_cfg_path, "r", encoding="utf-8") as f:
            a_cfg = yaml.safe_load(f)
        ai4i_summary = run_windowing_from_config(a_cfg, root_dir=root_dir)
        return cmapss_summary, ai4i_summary

    print("Executing Stage 7: Normalizing, collating .npz, and routing C-MAPSS FD001 windows...")
    cmapss_summary = process_cmapss_windows(root_dir=root_dir)
    print("Executing Stage 7: Normalizing, collating .npz, and routing AI4I 2020 static windows...")
    ai4i_summary = process_ai4i_windows(root_dir=root_dir)
    return cmapss_summary, ai4i_summary


if __name__ == "__main__":
    c_sum, a_sum = run_stage7_pipeline()
    print("\nStage 7 Execution Complete.")
    print("C-MAPSS FD001 Split Summary:")
    for k, v in c_sum.items():
        print(f"  {k}: {v['sample_count']} samples across {v['run_count']} runs (npz: {v['files']['canonical_data_npz']})")
    print("AI4I 2020 Split Summary:")
    for k, v in a_sum.items():
        print(f"  {k}: {v['sample_count']} samples across {v['run_count']} assets (npz: {v['files']['canonical_data_npz']})")
