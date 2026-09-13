"""Unit tests for train-only per-modality normalization module.

Validates:
- Scaler parameters are fitted strictly on train split without leakage
- Inverse-transform round-trips accurately
- Invariant / constant channels normalize to 0.0 with zero NaNs and zero Infs
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipelines.data.schema import ModalityMetadata
from pipelines.data.preprocessing.normalization import ModalityScaler, PerModalityStandardScaler


class TestNormalization:
    """Test suite for ModalityScaler and PerModalityStandardScaler."""

    def test_train_only_fitting_no_leakage(self):
        """Verifies that scaler parameters are computed exclusively from training data."""
        # Synthetic data: Train has mean=10.0, Test has mean=100.0
        train_df = pd.DataFrame({"sensor_a": np.array([8.0, 10.0, 12.0], dtype=np.float64)})
        test_df = pd.DataFrame({"sensor_a": np.array([90.0, 100.0, 110.0], dtype=np.float64)})

        scaler = ModalityScaler(modality_name="test_mod", channels=["sensor_a"])
        scaler.fit(train_df)

        assert scaler.channel_params["sensor_a"].mean == pytest.approx(10.0)
        # Verify transforming test data uses train mean (10.0), NOT test mean (100.0)
        norm_test = scaler.transform(test_df)
        expected_norm_100 = (100.0 - 10.0) / scaler.channel_params["sensor_a"].std
        assert norm_test[1, 0] == pytest.approx(expected_norm_100, rel=1e-4)

    def test_inverse_transform_roundtrip(self):
        """Verifies that transform followed by inverse_transform recovers original values."""
        rng = np.random.RandomState(42)
        raw_vals = rng.normal(loc=550.0, scale=35.0, size=(100, 3)).astype(np.float32)
        df = pd.DataFrame(raw_vals, columns=["ch1", "ch2", "ch3"])

        scaler = ModalityScaler(modality_name="vibration", channels=["ch1", "ch2", "ch3"])
        scaler.fit(df)

        norm_arr = scaler.transform(df)
        recovered_arr = scaler.inverse_transform(norm_arr)

        np.testing.assert_allclose(recovered_arr, raw_vals, atol=1e-4, err_msg="Inverse transform failed round-trip")

    def test_constant_channel_no_nan_or_inf(self):
        """Verifies that zero-variance invariant channels normalize to 0.0 without NaN/Inf."""
        # 50 rows of invariant sensor reading: constant 518.67
        const_df = pd.DataFrame({
            "invariant_s1": np.full(50, 518.67, dtype=np.float64),
            "varying_s2": np.linspace(10.0, 60.0, 50, dtype=np.float64),
        })

        scaler = ModalityScaler(modality_name="inlet", channels=["invariant_s1", "varying_s2"])
        scaler.fit(const_df)

        # Confirm parameter guard for constant channel
        param = scaler.channel_params["invariant_s1"]
        assert param.is_constant is True
        assert param.std == 1.0
        assert param.mean == pytest.approx(518.67)

        # Transform and verify
        normed = scaler.transform(const_df)
        assert not np.isnan(normed).any(), "NaN detected in normalized array"
        assert not np.isinf(normed).any(), "Inf detected in normalized array"
        np.testing.assert_allclose(normed[:, 0], 0.0, atol=1e-7, err_msg="Constant channel must normalize to exactly 0.0")

        # Inverse transform must recover original constant value
        recovered = scaler.inverse_transform(normed)
        np.testing.assert_allclose(recovered[:, 0], 518.67, atol=1e-5)

    def test_per_modality_standard_scaler_orchestration(self):
        """Verifies multi-modality orchestration across distinct modalities."""
        df = pd.DataFrame({
            "t1": [300.0, 310.0, 320.0],
            "p1": [14.7, 15.0, 15.3],
        })
        modalities = {
            "temp": ModalityMetadata(name="temp", channels=["t1"], units=["K"], description="Temp"),
            "press": ModalityMetadata(name="press", channels=["p1"], units=["psi"], description="Press"),
        }

        scaler = PerModalityStandardScaler(dataset_id="test_ds", scaler_version="1.0.0")
        scaler.fit(train_df=df, modalities_metadata=modalities, train_run_ids=["run1"], run_col="dummy")

        assert scaler.fitted is True
        assert "temp" in scaler.modalities
        assert "press" in scaler.modalities
        assert scaler.modalities["temp"].channel_params["t1"].mean == pytest.approx(310.0)
