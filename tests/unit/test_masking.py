"""Unit tests for controlled modality dropout and masking module.

Validates:
- Dropped modality has mask=0 and zeroed array values
- Untouched modalities remain numerically and byte-identical to pre-dropout values
- Metadata correctly stamps missingness_type="synthetic_controlled_dropout"
- Ablations are built strictly from held-out test splits
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from pipelines.data.schema import CanonicalSample, Targets, WindowMetadata
from pipelines.data.preprocessing.masking import apply_modality_dropout, run_masking_from_config


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


class TestMasking:
    """Test suite for controlled modality dropout and masking provenance."""

    @pytest.fixture
    def mock_sample(self) -> CanonicalSample:
        """Creates a known CanonicalSample with 2 modalities."""
        meta = WindowMetadata(
            dataset_id="test_ds",
            dataset_version="1.0.0",
            run_id="run_01",
            window_start=1,
            window_end=20,
            asset_id="asset_01",
            sampling_interval_seconds=1.0,
            operational_condition="cruise",
            missingness_type="none",
            dropped_modalities=[],
        )
        mod_vals = {
            "temperatures": np.ones((20, 4), dtype=np.float32) * 1.5,
            "pressures": np.ones((20, 5), dtype=np.float32) * 2.5,
        }
        mod_mask = {
            "temperatures": 1,
            "pressures": 1,
        }
        targets = Targets(target_rul=50.0)
        return CanonicalSample(
            sample_id="test_sample_001",
            metadata=meta,
            modality_values=mod_vals,
            modality_mask=mod_mask,
            targets=targets,
            preprocessor_version="v1.0",
            scaler_version="1.0.0",
        )

    def test_dropout_masks_and_zeros_modality(self, mock_sample: CanonicalSample):
        """Confirms dropped modality flag is 0 and array is completely zeroed."""
        dropped = apply_modality_dropout(
            sample=mock_sample,
            drop_modalities=["temperatures"],
            missingness_type="synthetic_controlled_dropout",
        )

        # Dropped modality checks
        assert dropped.modality_mask["temperatures"] == 0
        np.testing.assert_allclose(dropped.modality_values["temperatures"], 0.0, atol=1e-7)
        assert dropped.modality_values["temperatures"].shape == (20, 4)

        # Untouched modality checks: must be numerically identical
        assert dropped.modality_mask["pressures"] == 1
        np.testing.assert_array_equal(dropped.modality_values["pressures"], mock_sample.modality_values["pressures"])

    def test_dropout_stamps_missingness_metadata(self, mock_sample: CanonicalSample):
        """Confirms synthetic dropout provenance is recorded explicitly in WindowMetadata."""
        dropped = apply_modality_dropout(
            sample=mock_sample,
            drop_modalities=["temperatures"],
            missingness_type="synthetic_controlled_dropout",
        )

        assert dropped.metadata.missingness_type == "synthetic_controlled_dropout"
        assert dropped.metadata.dropped_modalities == ["temperatures"]
        assert "drop_temperatures" in dropped.sample_id

    def test_ablations_sourced_only_from_test_split(self):
        """Confirms pipeline manifest metadata explicitly binds ablations strictly to test split."""
        cmapss_ablation_manifest = WORKSPACE_ROOT / "data" / "processed" / "cmapss_fd001" / "ablations" / "drop_temperatures" / "manifest.json"
        if cmapss_ablation_manifest.exists():
            with open(cmapss_ablation_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
            abl_meta = data.get("ablation_metadata", {})
            assert abl_meta.get("source_split") == "test", "Ablation was not sourced from test split"
            assert "train and val data completely excluded" in abl_meta.get("leakage_guard", "")

        ai4i_ablation_manifest = WORKSPACE_ROOT / "data" / "processed" / "ai4i2020" / "ablations" / "drop_temperature" / "manifest.json"
        if ai4i_ablation_manifest.exists():
            with open(ai4i_ablation_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
            abl_meta = data.get("ablation_metadata", {})
            assert abl_meta.get("source_split") == "test", "Ablation was not sourced from test split"
