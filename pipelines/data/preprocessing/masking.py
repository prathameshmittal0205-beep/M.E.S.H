"""Missing-modality masking and controlled test-time dropout module for MESH.

Conforms to docs/model/MISSING_MODALITY_EXPERIMENTS.md, docs/model/MODEL_SPEC.md,
and docs/data/DATA_PIPELINE.md:
- Explicit missingness provenance: Every dropped modality records
  missingness_type="synthetic_controlled_dropout" (never mistaken for natural gaps).
- Authoritative mask semantics: modality_mask[m] = 0 is the primary signal for
  cross-attention negative bias. modality_values[m] is zeroed out for tensor shape
  integrity and to prevent unmasked data leakage.
- Held-out evaluation only: Ablation test suites are built strictly from the
  test split, never train or validation splits.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from pipelines.data.schema import CanonicalSample, WindowMetadata
from pipelines.data.adapters.cmapss import CMAPSSAdapter
from pipelines.data.adapters.ai4i import AI4IAdapter
from pipelines.data.preprocessing.windowing import save_canonical_split, verify_normalized_sample


def apply_modality_dropout(
    sample: CanonicalSample,
    drop_modalities: Sequence[str],
    missingness_type: str = "synthetic_controlled_dropout",
) -> CanonicalSample:
    """Applies controlled modality dropout to a CanonicalSample.

    Modality mask flags are set to 0 for dropped modalities.
    Values are zeroed out to ensure tensor regularity without data leakage.
    Metadata explicitly records the missingness type as synthetic controlled dropout.
    """
    new_mask = dict(sample.modality_mask)
    new_values = {}

    for mod_name, arr in sample.modality_values.items():
        if mod_name in drop_modalities:
            new_mask[mod_name] = 0
            new_values[mod_name] = np.zeros_like(arr, dtype=np.float32)
        else:
            new_values[mod_name] = arr.copy()

    # Create updated metadata recording synthetic dropout provenance
    orig_meta = sample.metadata
    new_meta = WindowMetadata(
        dataset_id=orig_meta.dataset_id,
        dataset_version=orig_meta.dataset_version,
        run_id=orig_meta.run_id,
        window_start=orig_meta.window_start,
        window_end=orig_meta.window_end,
        asset_id=orig_meta.asset_id,
        sampling_interval_seconds=orig_meta.sampling_interval_seconds,
        operational_condition=orig_meta.operational_condition,
        missingness_type=missingness_type,
        dropped_modalities=list(drop_modalities),
    )

    dropped_suffix = "_".join(drop_modalities)
    new_sample_id = f"{sample.sample_id}_drop_{dropped_suffix}"

    return CanonicalSample(
        sample_id=new_sample_id,
        metadata=new_meta,
        modality_values=new_values,
        modality_mask=new_mask,
        targets=sample.targets,
        preprocessor_version=sample.preprocessor_version,
        scaler_version=sample.scaler_version,
        static_features=dict(sample.static_features) if sample.static_features else None,
    )


def generate_single_modality_ablations(
    test_samples: Sequence[CanonicalSample],
    supported_modalities: Sequence[str],
    base_output_dir: Path,
    dataset_id: str,
    preprocessor_version: str,
    scaler_version: str,
    window_size: int,
    stride: Optional[int],
    modalities_metadata: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Generates per-modality drop test sets for test-time ablation studies.

    Strictly uses held-out test split samples only. Never train or val splits.
    """
    ablation_summaries: Dict[str, Dict[str, Any]] = {}
    ablations_root = base_output_dir / "ablations"
    ablations_root.mkdir(parents=True, exist_ok=True)

    # Extract distinct test run IDs for manifest tracking
    test_run_ids = sorted(list(set(s.metadata.run_id for s in test_samples)))

    for mod_to_drop in supported_modalities:
        split_name = f"drop_{mod_to_drop}"
        split_dir = ablations_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        dropped_samples: List[CanonicalSample] = []
        for s in test_samples:
            dropped_s = apply_modality_dropout(
                sample=s,
                drop_modalities=[mod_to_drop],
                missingness_type="synthetic_controlled_dropout",
            )
            verify_normalized_sample(dropped_s)
            dropped_samples.append(dropped_s)

        manifest_meta = save_canonical_split(
            samples=dropped_samples,
            split_dir=split_dir,
            dataset_id=dataset_id,
            split_name=split_name,
            window_size=window_size,
            stride=stride,
            preprocessor_version=preprocessor_version,
            scaler_version=scaler_version,
            run_ids=test_run_ids,
            modalities_metadata=modalities_metadata,
        )

        manifest_meta["ablation_metadata"] = {
            "ablation_protocol": "single_modality_test_dropout",
            "dropped_modality": mod_to_drop,
            "missingness_type": "synthetic_controlled_dropout",
            "source_split": "test",
            "leakage_guard": "Sourced exclusively from test split; train and val data completely excluded.",
        }

        # Update JSON manifest with ablation metadata
        manifest_json = split_dir / "manifest.json"
        with open(manifest_json, "w", encoding="utf-8") as f:
            json.dump(manifest_meta, f, indent=2)

        ablation_summaries[split_name] = manifest_meta

    return ablation_summaries


def run_masking_from_config(
    config: Dict[str, Any],
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generates controlled test-time modality dropout suites driven by configuration."""
    if root_dir is None:
        root_dir = Path.cwd()

    ds_cfg = config["dataset"]
    mask_cfg = config.get("masking", {})
    paths_cfg = config["paths"]

    dataset_name = ds_cfg["name"]
    source_split = mask_cfg.get("source_split", "test")

    processed_dir = root_dir / paths_cfg["processed_dir"]
    source_split_dir = processed_dir / source_split
    source_pkl = source_split_dir / "samples.pkl"
    source_manifest_path = source_split_dir / "manifest.json"

    if not source_pkl.exists():
        raise FileNotFoundError(f"Source split samples not found at {source_pkl}. Run windowing stage first.")
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Source split manifest not found at {source_manifest_path}. Run windowing stage first.")

    with open(source_pkl, "rb") as f:
        source_samples: List[CanonicalSample] = pickle.load(f)

    with open(source_manifest_path, "r", encoding="utf-8") as f:
        source_manifest = json.load(f)

    if dataset_name == "cmapss_fd001":
        subdataset = ds_cfg.get("subdataset", "FD001")
        drop_invariant = ds_cfg.get("drop_invariant_sensors", False)
        adapter = CMAPSSAdapter(subdataset=subdataset, drop_invariant_sensors=drop_invariant)
    elif dataset_name == "ai4i2020":
        adapter = AI4IAdapter(dataset_version=ds_cfg.get("dataset_version", "1.0.0"))
    else:
        raise ValueError(f"Unsupported dataset for masking: '{dataset_name}'")

    modalities = list(adapter.supported_modalities.keys())

    return generate_single_modality_ablations(
        test_samples=source_samples,
        supported_modalities=modalities,
        base_output_dir=processed_dir,
        dataset_id=adapter.dataset_id,
        preprocessor_version=source_manifest["preprocessor_version"],
        scaler_version=source_manifest["scaler_version"],
        window_size=source_manifest["window_size"],
        stride=source_manifest.get("stride"),
        modalities_metadata=adapter.supported_modalities,
    )


def process_cmapss_ablations(
    root_dir: Optional[Path] = None,
    subdataset: str = "FD001",
) -> Dict[str, Any]:
    """Generates all single-modality drop test sets for C-MAPSS FD001."""
    if root_dir is None:
        root_dir = Path.cwd()

    import yaml
    cfg_path = root_dir / "configs" / "data" / "cmapss_fd001.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return run_masking_from_config(cfg, root_dir=root_dir)

    adapter = CMAPSSAdapter(subdataset=subdataset)
    test_split_dir = root_dir / "data" / "processed" / f"cmapss_{subdataset.lower()}" / "test"
    test_pkl = test_split_dir / "samples.pkl"

    if not test_pkl.exists():
        raise FileNotFoundError(f"C-MAPSS test samples not found at {test_pkl}. Run Stage 7 first.")

    with open(test_pkl, "rb") as f:
        test_samples: List[CanonicalSample] = pickle.load(f)

    with open(test_split_dir / "manifest.json", "r", encoding="utf-8") as f:
        test_manifest = json.load(f)

    modalities = list(adapter.supported_modalities.keys())
    base_out = root_dir / "data" / "processed" / f"cmapss_{subdataset.lower()}"

    return generate_single_modality_ablations(
        test_samples=test_samples,
        supported_modalities=modalities,
        base_output_dir=base_out,
        dataset_id=adapter.dataset_id,
        preprocessor_version=test_manifest["preprocessor_version"],
        scaler_version=test_manifest["scaler_version"],
        window_size=test_manifest["window_size"],
        stride=test_manifest.get("stride", 5),
        modalities_metadata=adapter.supported_modalities,
    )


def process_ai4i_ablations(
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generates all single-modality drop test sets for AI4I 2020."""
    if root_dir is None:
        root_dir = Path.cwd()

    import yaml
    cfg_path = root_dir / "configs" / "data" / "ai4i2020.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return run_masking_from_config(cfg, root_dir=root_dir)

    adapter = AI4IAdapter()
    test_split_dir = root_dir / "data" / "processed" / "ai4i2020" / "test"
    test_pkl = test_split_dir / "samples.pkl"

    if not test_pkl.exists():
        raise FileNotFoundError(f"AI4I test samples not found at {test_pkl}. Run Stage 7 first.")

    with open(test_pkl, "rb") as f:
        test_samples: List[CanonicalSample] = pickle.load(f)

    with open(test_split_dir / "manifest.json", "r", encoding="utf-8") as f:
        test_manifest = json.load(f)

    modalities = list(adapter.supported_modalities.keys())
    base_out = root_dir / "data" / "processed" / "ai4i2020"

    return generate_single_modality_ablations(
        test_samples=test_samples,
        supported_modalities=modalities,
        base_output_dir=base_out,
        dataset_id=adapter.dataset_id,
        preprocessor_version=test_manifest["preprocessor_version"],
        scaler_version=test_manifest["scaler_version"],
        window_size=1,
        stride=None,
        modalities_metadata=adapter.supported_modalities,
    )


def run_stage8_pipeline(root_dir: Optional[Path] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Executes full Stage 8 ablation test-set generation for both datasets."""
    if root_dir is None:
        root_dir = Path.cwd()

    import yaml
    c_cfg_path = root_dir / "configs" / "data" / "cmapss_fd001.yaml"
    a_cfg_path = root_dir / "configs" / "data" / "ai4i2020.yaml"

    if c_cfg_path.exists() and a_cfg_path.exists():
        print("Executing Stage 8 via configs: Generating C-MAPSS FD001 single-modality drop test sets...")
        with open(c_cfg_path, "r", encoding="utf-8") as f:
            c_cfg = yaml.safe_load(f)
        cmapss_ablations = run_masking_from_config(c_cfg, root_dir=root_dir)

        print("Executing Stage 8 via configs: Generating AI4I 2020 single-modality drop test sets...")
        with open(a_cfg_path, "r", encoding="utf-8") as f:
            a_cfg = yaml.safe_load(f)
        ai4i_ablations = run_masking_from_config(a_cfg, root_dir=root_dir)
        return cmapss_ablations, ai4i_ablations

    print("Executing Stage 8: Generating C-MAPSS FD001 single-modality drop test sets...")
    cmapss_ablations = process_cmapss_ablations(root_dir=root_dir)
    print("Executing Stage 8: Generating AI4I 2020 single-modality drop test sets...")
    ai4i_ablations = process_ai4i_ablations(root_dir=root_dir)
    return cmapss_ablations, ai4i_ablations


if __name__ == "__main__":
    c_abl, a_abl = run_stage8_pipeline()
    print("\nStage 8 Execution Complete.")
    print(f"C-MAPSS FD001 Ablation Suites ({len(c_abl)} modalities):")
    for k, v in c_abl.items():
        print(f"  {k}: {v['sample_count']} samples (missingness_type='{v['ablation_metadata']['missingness_type']}')")
    print(f"AI4I 2020 Ablation Suites ({len(a_abl)} modalities):")
    for k, v in a_abl.items():
        print(f"  {k}: {v['sample_count']} samples (missingness_type='{v['ablation_metadata']['missingness_type']}')")
