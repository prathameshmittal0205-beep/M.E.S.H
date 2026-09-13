"""Asset- and run-level split management for MESH.

Ensures zero data leakage between train, validation, and test sets
per docs/data/DATA_PIPELINE.md, docs/team/NAMAN_DATA.md, and docs/model/TRAINING_EVALUATION.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pipelines.data.schema import SplitManifest


def create_cmapss_split(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    subdataset: str = "FD001",
    dataset_version: str = "1.0.0",
    output_path: Optional[Path] = None,
) -> SplitManifest:
    """Creates a strictly leak-free asset-level split for NASA C-MAPSS.

    Engines are partitioned by unit_number. All cycles and temporal windows
    belonging to an engine remain strictly inside that engine's assigned set.
    """
    if "unit_number" not in train_df.columns:
        raise ValueError("train_df must contain 'unit_number' column")

    unique_units = sorted(train_df["unit_number"].unique().tolist())
    n_units = len(unique_units)

    # Deterministic permutation of engine unit IDs
    rng = np.random.RandomState(seed)
    shuffled_units = unique_units.copy()
    rng.shuffle(shuffled_units)

    n_val = int(round(n_units * val_ratio))
    val_units = sorted(shuffled_units[:n_val])
    train_units = sorted(shuffled_units[n_val:])

    # Format run IDs as strings e.g. "unit_001"
    train_runs = [f"unit_{u:03d}" for u in train_units]
    val_runs = [f"unit_{u:03d}" for u in val_units]

    # Official held-out test engines: strictly read from verified source dataframe
    if test_df is None or "unit_number" not in test_df.columns:
        raise ValueError(
            "test_df must be provided and contain 'unit_number' to determine real test engine runs. "
            "Synthesized placeholder engine ranges are strictly forbidden per docs/NO_HALLUCINATION_POLICY.md."
        )
    test_units = sorted(test_df["unit_number"].unique().tolist())
    test_runs = [f"test_unit_{u:03d}" for u in test_units]

    manifest = SplitManifest(
        dataset_id=f"nasa_cmapss_{subdataset.lower()}",
        dataset_version=dataset_version,
        split_strategy=f"asset_level_engine_unit_holdout_{int((1 - val_ratio)*100)}_{int(val_ratio*100)}",
        train_runs=train_runs,
        val_runs=val_runs,
        test_runs=test_runs,
        leakage_verified=False,
        created_at=datetime.now(timezone.utc).isoformat(),
        notes=f"Split by unit_number with seed={seed}. Train engines: {len(train_runs)}, Val engines: {len(val_runs)}, Test engines: {len(test_runs)}.",
    )

    # Verify zero leakage across engine sets
    manifest.verify_no_overlap()
    # If passed without exception, mark leakage_verified True
    manifest = SplitManifest(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        split_strategy=manifest.split_strategy,
        train_runs=manifest.train_runs,
        val_runs=manifest.val_runs,
        test_runs=manifest.test_runs,
        leakage_verified=True,
        created_at=manifest.created_at,
        notes=manifest.notes,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)

    return manifest


def create_ai4i_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_col: Optional[str] = "Machine failure",
    dataset_version: str = "1.0.0",
    output_path: Optional[Path] = None,
) -> SplitManifest:
    """Creates an asset-level split for AI4I 2020 synthetic benchmark.

    Because AI4I consists of 10,000 independent single-snapshot machines
    (10,000 unique Product IDs with 1 observation per machine), splitting
    by Product ID partitions distinct machines into mutually disjoint subsets.
    Stratification preserves rare failure mode proportions across splits.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    product_ids = df["Product ID"].values
    strat_labels = df[stratify_col].values if stratify_col and stratify_col in df.columns else None

    # Step 1: Split off test set
    train_val_ids, test_ids, y_train_val, _ = train_test_split(
        product_ids,
        strat_labels,
        test_size=test_ratio,
        random_state=seed,
        stratify=strat_labels,
    )

    # Step 2: Split train and validation sets
    relative_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids,
        y_train_val,
        test_size=relative_val_ratio,
        random_state=seed,
        stratify=y_train_val,
    )

    train_runs = [str(pid) for pid in sorted(train_ids)]
    val_runs = [str(pid) for pid in sorted(val_ids)]
    test_runs = [str(pid) for pid in sorted(test_ids)]

    manifest = SplitManifest(
        dataset_id="ai4i2020",
        dataset_version=dataset_version,
        split_strategy=f"asset_stratified_{int(train_ratio*100)}_{int(val_ratio*100)}_{int(test_ratio*100)}",
        train_runs=train_runs,
        val_runs=val_runs,
        test_runs=test_runs,
        leakage_verified=False,
        created_at=datetime.now(timezone.utc).isoformat(),
        notes=(
            f"Asset-level split by unique Product ID (1 observation per asset) with seed={seed}. "
            f"Stratified on '{stratify_col}'. Train assets: {len(train_runs)}, "
            f"Val assets: {len(val_runs)}, Test assets: {len(test_runs)}."
        ),
    )

    # Verify zero overlap
    manifest.verify_no_overlap()
    manifest = SplitManifest(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        split_strategy=manifest.split_strategy,
        train_runs=manifest.train_runs,
        val_runs=manifest.val_runs,
        test_runs=manifest.test_runs,
        leakage_verified=True,
        created_at=manifest.created_at,
        notes=manifest.notes,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)

    return manifest


def run_split_from_config(
    config: Dict[str, Any],
    root_dir: Optional[Path] = None,
) -> SplitManifest:
    """Creates a leak-free asset-level split driven entirely by a configuration dictionary."""
    if root_dir is None:
        root_dir = Path.cwd()

    ds_cfg = config["dataset"]
    split_cfg = config["split"]
    paths_cfg = config["paths"]

    dataset_name = ds_cfg["name"]
    output_path = root_dir / paths_cfg["split_output"]

    if dataset_name == "cmapss_fd001":
        from pipelines.data.adapters.cmapss import CMAPSSAdapter
        subdataset = ds_cfg.get("subdataset", "FD001")
        drop_invariant = ds_cfg.get("drop_invariant_sensors", False)
        adapter = CMAPSSAdapter(subdataset=subdataset, drop_invariant_sensors=drop_invariant)

        train_path = root_dir / paths_cfg["train_file"]
        test_path = root_dir / paths_cfg["test_file"] if paths_cfg.get("test_file") else None

        if not train_path.exists():
            raise FileNotFoundError(f"C-MAPSS training file not found: {train_path}")

        train_df = adapter.load_raw(train_path)
        test_df = adapter.load_raw(test_path) if test_path and test_path.exists() else None

        return create_cmapss_split(
            train_df=train_df,
            test_df=test_df,
            val_ratio=split_cfg["val_ratio"],
            seed=split_cfg["seed"],
            subdataset=subdataset,
            dataset_version=ds_cfg.get("dataset_version", "1.0.0"),
            output_path=output_path,
        )

    elif dataset_name == "ai4i2020":
        from pipelines.data.adapters.ai4i import AI4IAdapter
        adapter = AI4IAdapter(dataset_version=ds_cfg.get("dataset_version", "1.0.0"))

        raw_path = root_dir / paths_cfg["train_file"]
        if not raw_path.exists():
            raise FileNotFoundError(f"AI4I raw data file not found: {raw_path}")

        df = adapter.load_raw(raw_path)

        return create_ai4i_split(
            df=df,
            train_ratio=split_cfg["train_ratio"],
            val_ratio=split_cfg["val_ratio"],
            test_ratio=split_cfg["test_ratio"],
            seed=split_cfg["seed"],
            stratify_col=split_cfg.get("stratify_col", "Machine failure"),
            dataset_version=ds_cfg.get("dataset_version", "1.0.0"),
            output_path=output_path,
        )

    else:
        raise ValueError(f"Unsupported dataset for splitting: '{dataset_name}'")


def generate_all_splits(
    output_dir: Optional[Path] = None,
    seed: int = 42,
) -> Tuple[SplitManifest, SplitManifest]:
    """Generates and saves leak-free split manifests for C-MAPSS and AI4I 2020 via config."""
    import yaml

    root_dir = Path.cwd()
    cmapss_cfg_path = root_dir / "configs" / "data" / "cmapss_fd001.yaml"
    ai4i_cfg_path = root_dir / "configs" / "data" / "ai4i2020.yaml"

    if cmapss_cfg_path.exists() and ai4i_cfg_path.exists():
        with open(cmapss_cfg_path, "r", encoding="utf-8") as f:
            cmapss_cfg = yaml.safe_load(f)
        with open(ai4i_cfg_path, "r", encoding="utf-8") as f:
            ai4i_cfg = yaml.safe_load(f)

        cmapss_manifest = run_split_from_config(cmapss_cfg, root_dir=root_dir)
        ai4i_manifest = run_split_from_config(ai4i_cfg, root_dir=root_dir)
        return cmapss_manifest, ai4i_manifest

    # Fallback to direct adapter loading if configs do not exist
    from pipelines.data.adapters.cmapss import CMAPSSAdapter
    from pipelines.data.adapters.ai4i import AI4IAdapter

    if output_dir is None:
        output_dir = root_dir / "data" / "processed" / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmapss_adapter = CMAPSSAdapter(subdataset="FD001")
    train_path = root_dir / "data" / "raw" / "cmapss" / "train_FD001.txt"
    test_path = root_dir / "data" / "raw" / "cmapss" / "test_FD001.txt"
    train_df = cmapss_adapter.load_raw(train_path)
    test_df = cmapss_adapter.load_raw(test_path) if test_path.exists() else None

    cmapss_manifest = create_cmapss_split(
        train_df=train_df,
        test_df=test_df,
        val_ratio=0.20,
        seed=seed,
        subdataset="FD001",
        output_path=output_dir / "cmapss_fd001_split.json",
    )

    ai4i_adapter = AI4IAdapter()
    ai4i_path = root_dir / "data" / "raw" / "ai4i2020" / "ai4i2020.csv"
    ai4i_df = ai4i_adapter.load_raw(ai4i_path)

    ai4i_manifest = create_ai4i_split(
        df=ai4i_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
        stratify_col="Machine failure",
        output_path=output_dir / "ai4i2020_split.json",
    )

    return cmapss_manifest, ai4i_manifest


if __name__ == "__main__":
    cmapss_m, ai4i_m = generate_all_splits()
    print("Split generation complete.")
    print(f"C-MAPSS {cmapss_m.dataset_id}: Train={len(cmapss_m.train_runs)}, Val={len(cmapss_m.val_runs)}, Test={len(cmapss_m.test_runs)}, Leak-free={cmapss_m.leakage_verified}")
    print(f"AI4I 2020: Train={len(ai4i_m.train_runs)}, Val={len(ai4i_m.val_runs)}, Test={len(ai4i_m.test_runs)}, Leak-free={ai4i_m.leakage_verified}")

