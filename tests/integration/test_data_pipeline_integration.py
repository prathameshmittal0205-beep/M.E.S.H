"""End-to-end integration tests for MESH data pipeline.

Validates:
- Full CLI orchestration end-to-end into an isolated temporary directory
- Strict zero split leakage verification via SplitManifest.verify_no_overlap()
- Determinism guarantee: same seed and config produce identical .npz arrays across repeated runs
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest
import yaml

from pipelines.data.schema import SplitManifest
from pipelines.data.pipeline import PipelineOrchestrator, load_config


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


class TestPipelineIntegration:
    """Integration test suite executing the data pipeline end-to-end."""

    def test_pipeline_cmapss_end_to_end_in_temp_dir(self):
        """Runs the complete C-MAPSS pipeline in an isolated temp directory and verifies zero leakage."""
        cfg_path = WORKSPACE_ROOT / "configs" / "data" / "cmapss_fd001.yaml"
        config = load_config(cfg_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            # Route all outputs strictly into temp directory (no pollution of real data/)
            config_temp = copy.deepcopy(config)
            config_temp["paths"]["provenance_output"] = str(temp_root / "provenance.json")
            config_temp["paths"]["qa_report_output"] = str(temp_root / "qa_report.json")
            config_temp["paths"]["split_output"] = str(temp_root / "splits" / "cmapss_split.json")
            config_temp["paths"]["scaler_json_output"] = str(temp_root / "scalers" / "scaler.json")
            config_temp["paths"]["scaler_pkl_output"] = str(temp_root / "scalers" / "scaler.pkl")
            config_temp["paths"]["processed_dir"] = str(temp_root / "processed")

            orchestrator = PipelineOrchestrator(config=config_temp, root_dir=WORKSPACE_ROOT)
            orchestrator.run_all(clean=False)

            # 1. Verify split manifest zero leakage
            split_file = Path(config_temp["paths"]["split_output"])
            assert split_file.exists()
            with open(split_file, "r", encoding="utf-8") as f:
                split_dict = json.load(f)
            manifest = SplitManifest(**split_dict)
            manifest.verify_no_overlap()  # Must not raise
            assert manifest.leakage_verified is True

            # 2. Verify deliverable splits exist and contain valid data
            processed_dir = Path(config_temp["paths"]["processed_dir"])
            for split_name in ["train", "val", "test"]:
                npz_p = processed_dir / split_name / "data.npz"
                man_p = processed_dir / split_name / "manifest.json"
                assert npz_p.exists()
                assert man_p.exists()
                with np.load(npz_p, allow_pickle=False) as npz:
                    assert len(npz["sample_ids"]) > 0

    def test_pipeline_ai4i_end_to_end_in_temp_dir(self):
        """Runs the complete AI4I 2020 pipeline in an isolated temp directory and verifies zero leakage."""
        cfg_path = WORKSPACE_ROOT / "configs" / "data" / "ai4i2020.yaml"
        config = load_config(cfg_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            config_temp = copy.deepcopy(config)
            config_temp["paths"]["provenance_output"] = str(temp_root / "provenance.json")
            config_temp["paths"]["qa_report_output"] = str(temp_root / "qa_report.json")
            config_temp["paths"]["split_output"] = str(temp_root / "splits" / "ai4i_split.json")
            config_temp["paths"]["scaler_json_output"] = str(temp_root / "scalers" / "scaler.json")
            config_temp["paths"]["scaler_pkl_output"] = str(temp_root / "scalers" / "scaler.pkl")
            config_temp["paths"]["processed_dir"] = str(temp_root / "processed")

            orchestrator = PipelineOrchestrator(config=config_temp, root_dir=WORKSPACE_ROOT)
            orchestrator.run_all(clean=False)

            # Verify split manifest zero leakage
            split_file = Path(config_temp["paths"]["split_output"])
            assert split_file.exists()
            with open(split_file, "r", encoding="utf-8") as f:
                split_dict = json.load(f)
            manifest = SplitManifest(**split_dict)
            manifest.verify_no_overlap()
            assert manifest.leakage_verified is True

    def test_pipeline_determinism_across_runs(self):
        """Verifies that two independent runs with identical config produce 100% numerically equal arrays."""
        cfg_path = WORKSPACE_ROOT / "configs" / "data" / "cmapss_fd001.yaml"
        config = load_config(cfg_path)

        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            root1 = Path(tmpdir1)
            root2 = Path(tmpdir2)

            def build_cfg(root: Path):
                cfg = copy.deepcopy(config)
                cfg["paths"]["provenance_output"] = str(root / "provenance.json")
                cfg["paths"]["qa_report_output"] = str(root / "qa.json")
                cfg["paths"]["split_output"] = str(root / "split.json")
                cfg["paths"]["scaler_json_output"] = str(root / "scaler.json")
                cfg["paths"]["scaler_pkl_output"] = str(root / "scaler.pkl")
                cfg["paths"]["processed_dir"] = str(root / "processed")
                return cfg

            # Run 1
            orch1 = PipelineOrchestrator(config=build_cfg(root1), root_dir=WORKSPACE_ROOT)
            orch1.run_all()

            # Run 2
            orch2 = PipelineOrchestrator(config=build_cfg(root2), root_dir=WORKSPACE_ROOT)
            orch2.run_all()

            # Compare generated test .npz files array by array
            npz1_path = root1 / "processed" / "test" / "data.npz"
            npz2_path = root2 / "processed" / "test" / "data.npz"

            with np.load(npz1_path, allow_pickle=False) as data1, np.load(npz2_path, allow_pickle=False) as data2:
                assert set(data1.files) == set(data2.files)
                for k in data1.files:
                    arr1 = data1[k]
                    arr2 = data2[k]
                    np.testing.assert_array_equal(arr1, arr2, err_msg=f"Array '{k}' differed between identical pipeline runs")
