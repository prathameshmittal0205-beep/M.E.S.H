"""End-to-End Data Pipeline Orchestrator for MESH.

Orchestrates the full data preparation pipeline strictly from configuration:
  1. Provenance: Raw file scanning, SHA-256 checksumming, catalog generation
  2. Validation: Structural integrity, physical bounds audit, hard QA gate
  3. Splitting: Asset-level leak-free partitioning (train / val / test)
  4. Normalization: Train-only per-modality z-score scaler fitting
  5. Windowing: Sliding/static window slicing, scaler transform, .npz collation
  6. Masking: Test-time single-modality controlled dropout ablation suites

Usage:
  python -m pipelines.data.pipeline --config configs/data/cmapss_fd001.yaml
  python -m pipelines.data.pipeline --config configs/data/ai4i2020.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import yaml
import numpy as np

from pipelines.data.provenance import run_provenance_from_config
from pipelines.data.validation import run_validation_from_config
from pipelines.data.preprocessing.splitting import run_split_from_config
from pipelines.data.preprocessing.normalization import run_normalization_from_config
from pipelines.data.preprocessing.windowing import run_windowing_from_config
from pipelines.data.preprocessing.masking import run_masking_from_config


def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads and validates pipeline YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config format in {config_path}: expected dictionary root.")

    # Validate mandatory top-level schema sections
    required_sections = ["dataset", "paths", "split", "normalization", "windowing", "masking"]
    for sec in required_sections:
        if sec not in config:
            raise KeyError(f"Missing mandatory configuration section '{sec}' in {config_path}")

    return config


class PipelineOrchestrator:
    """Manages sequential execution of MESH data pipeline stages with strict fail-loud gates."""

    def __init__(self, config: Dict[str, Any], root_dir: Optional[Path] = None):
        self.config = config
        self.root_dir = root_dir.resolve() if root_dir else Path.cwd().resolve()
        self.dataset_name = self.config["dataset"]["name"]
        self.dataset_id = self.config["dataset"]["dataset_id"]

    def _resolve(self, relative_path: str) -> Path:
        """Resolves relative workspace path against root directory."""
        return (self.root_dir / relative_path).resolve()

    def stage_1_provenance(self) -> Dict[str, Any]:
        """Stage 1: Provenance Tracking & Verification."""
        print("\n" + "=" * 80)
        print(f"STAGE 1: Provenance Tracking & Verification [{self.dataset_id}]")
        print("=" * 80)

        raw_dir = self._resolve(self.config["paths"]["raw_dir"])
        if not raw_dir.exists():
            raise FileNotFoundError(f"[Stage 1 FAIL] Raw data directory missing: {raw_dir}")

        train_file = self._resolve(self.config["paths"]["train_file"])
        if not train_file.exists():
            raise FileNotFoundError(f"[Stage 1 FAIL] Mandatory raw train file missing: {train_file}")

        catalog = run_provenance_from_config(self.config, root_dir=self.root_dir)
        provenance_out = self._resolve(self.config["paths"]["provenance_output"])

        if not provenance_out.exists() or provenance_out.stat().st_size == 0:
            raise RuntimeError(f"[Stage 1 FAIL] Provenance catalog failed to generate at {provenance_out}")

        print(f"  [PASS] Provenance catalog written: {provenance_out}")
        print(f"  [PASS] Datasets cataloged: {list(catalog.get('datasets', {}).keys())}")
        return catalog

    def stage_2_validation(self) -> Dict[str, Any]:
        """Stage 2: Data Quality & Physical Bounds Validation Gate."""
        print("\n" + "=" * 80)
        print(f"STAGE 2: Data Quality & Physical Bounds Validation Gate [{self.dataset_id}]")
        print("=" * 80)

        # Fail-loud check on prerequisite raw data
        train_file = self._resolve(self.config["paths"]["train_file"])
        if not train_file.exists():
            raise FileNotFoundError(f"[Stage 2 FAIL] Raw data file missing: {train_file}")

        report, details = run_validation_from_config(self.config, root_dir=self.root_dir)
        qa_out = self._resolve(self.config["paths"]["qa_report_output"])

        if not qa_out.exists():
            raise RuntimeError(f"[Stage 2 FAIL] QA report was not saved to {qa_out}")

        # Enforce Hard QA Gate
        if not report.passed_qa:
            raise RuntimeError(
                f"[Stage 2 FAIL] Hard QA gate failed for {self.dataset_id}!\n"
                f"Missing values: {report.missing_value_count}, Duplicates: {report.duplicate_records_count}, "
                f"Monotonicity violations: {report.monotonicity_violations}, Invalid readings: {report.invalid_sensor_readings}.\n"
                f"Notes: {report.notes}"
            )

        print(f"  [PASS] Data quality audit passed QA gate: {qa_out}")
        print(f"  [PASS] Total records: {report.total_records:,} across {report.total_runs:,} assets/runs")
        print(f"  [PASS] Missing/NaN/Inf: {report.missing_value_count}, Duplicates: {report.duplicate_records_count}, Monotonicity: {report.monotonicity_violations}")
        return details

    def stage_3_splitting(self) -> Dict[str, Any]:
        """Stage 3: Asset-Level Leak-Free Split Generation."""
        print("\n" + "=" * 80)
        print(f"STAGE 3: Asset-Level Leak-Free Split Generation [{self.dataset_id}]")
        print("=" * 80)

        # Fail-loud check on prerequisite QA report
        qa_out = self._resolve(self.config["paths"]["qa_report_output"])
        if not qa_out.exists():
            raise FileNotFoundError(f"[Stage 3 FAIL] Prerequisite Stage 2 QA report missing at {qa_out}. Run Stage 2 first.")

        manifest = run_split_from_config(self.config, root_dir=self.root_dir)
        split_out = self._resolve(self.config["paths"]["split_output"])

        if not split_out.exists():
            raise RuntimeError(f"[Stage 3 FAIL] Split manifest was not saved to {split_out}")

        if not manifest.leakage_verified:
            raise RuntimeError(f"[Stage 3 FAIL] Data leakage verification failed for split {split_out}!")

        print(f"  [PASS] Split manifest created: {split_out}")
        print(f"  [PASS] Strategy: {manifest.split_strategy}")
        print(f"  [PASS] Train runs: {len(manifest.train_runs):,}, Val runs: {len(manifest.val_runs):,}, Test runs: {len(manifest.test_runs):,}")
        print(f"  [PASS] Zero asset/run overlap verified across all partitions.")
        return asdict_manifest(manifest)

    def stage_4_normalization(self) -> Any:
        """Stage 4: Train-Only Per-Modality Normalization Fitting."""
        print("\n" + "=" * 80)
        print(f"STAGE 4: Train-Only Per-Modality Normalization Fitting [{self.dataset_id}]")
        print("=" * 80)

        # Fail-loud check on prerequisite split manifest
        split_out = self._resolve(self.config["paths"]["split_output"])
        if not split_out.exists():
            raise FileNotFoundError(f"[Stage 4 FAIL] Prerequisite Stage 3 split manifest missing at {split_out}. Run Stage 3 first.")

        scaler = run_normalization_from_config(self.config, root_dir=self.root_dir)
        scaler_json = self._resolve(self.config["paths"]["scaler_json_output"])
        scaler_pkl = self._resolve(self.config["paths"]["scaler_pkl_output"])

        if not scaler_json.exists():
            raise RuntimeError(f"[Stage 4 FAIL] Scaler JSON parameters missing at {scaler_json}")
        if not scaler_pkl.exists():
            raise RuntimeError(f"[Stage 4 FAIL] Scaler pickle missing at {scaler_pkl}")
        if not scaler.fitted:
            raise RuntimeError(f"[Stage 4 FAIL] Scaler fitted flag is False.")

        print(f"  [PASS] Scaler fitted strictly on train split: {scaler_pkl}")
        print(f"  [PASS] Auditable parameter JSON saved: {scaler_json}")
        print(f"  [PASS] Fitted modalities: {list(scaler.modalities.keys())}")
        return scaler

    def stage_5_windowing(self) -> Dict[str, Any]:
        """Stage 5: Window Extraction, Normalization Transform & Collation (data.npz + manifest.json)."""
        print("\n" + "=" * 80)
        print(f"STAGE 5: Window Extraction, Normalization & Collation [{self.dataset_id}]")
        print("=" * 80)

        # Fail-loud check on prerequisite split and scaler
        split_out = self._resolve(self.config["paths"]["split_output"])
        scaler_pkl = self._resolve(self.config["paths"]["scaler_pkl_output"])
        if not split_out.exists():
            raise FileNotFoundError(f"[Stage 5 FAIL] Prerequisite split manifest missing at {split_out}")
        if not scaler_pkl.exists():
            raise FileNotFoundError(f"[Stage 5 FAIL] Prerequisite scaler missing at {scaler_pkl}")

        summary = run_windowing_from_config(self.config, root_dir=self.root_dir)
        processed_dir = self._resolve(self.config["paths"]["processed_dir"])

        for split_name in ["train", "val", "test"]:
            split_dir = processed_dir / split_name
            npz_path = split_dir / "data.npz"
            manifest_path = split_dir / "manifest.json"

            if not npz_path.exists():
                raise RuntimeError(f"[Stage 5 FAIL] Missing canonical {npz_path}")
            if not manifest_path.exists():
                raise RuntimeError(f"[Stage 5 FAIL] Missing split manifest {manifest_path}")

            with np.load(npz_path, allow_pickle=False) as npz:
                sample_count = len(npz["sample_ids"])
                if sample_count == 0:
                    raise ValueError(f"[Stage 5 FAIL] {split_name} split in {npz_path} contains 0 samples!")

            split_meta = summary[split_name]
            print(f"  [PASS] Split '{split_name}': {split_meta['sample_count']:,} samples across {split_meta['run_count']:,} assets (file: {npz_path.name})")

        return summary

    def stage_6_masking(self) -> Dict[str, Any]:
        """Stage 6: Controlled Test-Time Modality Dropout Ablations."""
        print("\n" + "=" * 80)
        print(f"STAGE 6: Controlled Modality Dropout Ablations [{self.dataset_id}]")
        print("=" * 80)

        # Fail-loud check on prerequisite test split
        processed_dir = self._resolve(self.config["paths"]["processed_dir"])
        test_dir = processed_dir / "test"
        test_npz = test_dir / "data.npz"
        test_pkl = test_dir / "samples.pkl"

        if not test_npz.exists() or not test_pkl.exists():
            raise FileNotFoundError(
                f"[Stage 6 FAIL] Prerequisite Stage 5 test artifacts missing at {test_dir}. Run Stage 5 first."
            )

        with np.load(test_npz, allow_pickle=False) as data:
            expected_test_samples = len(data["sample_ids"])

        ablations = run_masking_from_config(self.config, root_dir=self.root_dir)
        ablations_dir = processed_dir / "ablations"

        if not ablations_dir.exists():
            raise RuntimeError(f"[Stage 6 FAIL] Ablations root directory missing: {ablations_dir}")

        for mod_drop_name, meta in ablations.items():
            mod_dir = ablations_dir / mod_drop_name
            npz_path = mod_dir / "data.npz"
            manifest_path = mod_dir / "manifest.json"

            if not npz_path.exists():
                raise RuntimeError(f"[Stage 6 FAIL] Missing ablation data array: {npz_path}")
            if not manifest_path.exists():
                raise RuntimeError(f"[Stage 6 FAIL] Missing ablation manifest: {manifest_path}")

            with np.load(npz_path, allow_pickle=False) as npz:
                count = len(npz["sample_ids"])
                if count != expected_test_samples:
                    raise ValueError(
                        f"[Stage 6 FAIL] Ablation {mod_drop_name} sample count ({count}) does not match test set ({expected_test_samples})"
                    )

            print(f"  [PASS] Ablation '{mod_drop_name}': {meta['sample_count']:,} samples (missingness_type: {meta['ablation_metadata']['missingness_type']})")

        return ablations

    def run_all(self, clean: bool = False) -> None:
        """Executes all pipeline stages sequentially with timing and verification."""
        t_start = time.time()
        print("\n" + "#" * 80)
        print(f"STARTING MESH DATA PIPELINE: {self.dataset_id} (Config: {self.dataset_name})")
        print("#" * 80)

        if clean:
            processed_dir = self._resolve(self.config["paths"]["processed_dir"])
            if processed_dir.exists():
                print(f"Cleaning existing processed directory: {processed_dir}")
                shutil.rmtree(processed_dir)

        # Sequential orchestration: provenance -> validation -> split -> normalize -> window -> mask
        self.stage_1_provenance()
        self.stage_2_validation()
        self.stage_3_splitting()
        self.stage_4_normalization()
        self.stage_5_windowing()
        self.stage_6_masking()

        total_elapsed = time.time() - t_start
        print("\n" + "#" * 80)
        print(f"PIPELINE COMPLETE: {self.dataset_id} in {total_elapsed:.2f} seconds.")
        print(f"Deliverable directory: {self._resolve(self.config['paths']['processed_dir'])}")
        print("#" * 80 + "\n")


def asdict_manifest(manifest: Any) -> Dict[str, Any]:
    from dataclasses import asdict
    return asdict(manifest)


def main():
    parser = argparse.ArgumentParser(
        description="MESH End-to-End Data Pipeline CLI Entrypoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g. configs/data/cmapss_fd001.yaml)",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Project workspace root directory (defaults to current working directory)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe dataset processed output directory before running",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "provenance", "validation", "split", "normalize", "window", "mask"],
        help="Specific pipeline stage to run",
    )

    args = parser.parse_args()
    config_path = Path(args.config)
    root_path = Path(args.root_dir) if args.root_dir else Path.cwd()

    config = load_config(config_path)
    orchestrator = PipelineOrchestrator(config=config, root_dir=root_path)

    if args.stage == "all":
        orchestrator.run_all(clean=args.clean)
    elif args.stage == "provenance":
        orchestrator.stage_1_provenance()
    elif args.stage == "validation":
        orchestrator.stage_2_validation()
    elif args.stage == "split":
        orchestrator.stage_3_splitting()
    elif args.stage == "normalize":
        orchestrator.stage_4_normalization()
    elif args.stage == "window":
        orchestrator.stage_5_windowing()
    elif args.stage == "mask":
        orchestrator.stage_6_masking()


if __name__ == "__main__":
    main()
