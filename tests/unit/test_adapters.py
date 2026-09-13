"""Unit tests for dataset adapters (C-MAPSS and AI4I 2020).

Validates:
- Parser correctness and trajectory grouping
- Temporal cycle and UDI monotonicity
- RUL formula correctness for both training (t_max - t) and test (RUL_final + t_max - t)
- Hard rejection of window_size > 1 on AI4I 2020 tabular single-snapshot data
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipelines.data.adapters.cmapss import CMAPSSAdapter
from pipelines.data.adapters.ai4i import AI4IAdapter


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


class TestCMAPSSAdapter:
    """Test suite for NASA C-MAPSS Turbofan Engine Adapter."""

    @pytest.fixture
    def adapter(self) -> CMAPSSAdapter:
        return CMAPSSAdapter(subdataset="FD001", drop_invariant_sensors=False)

    @pytest.fixture
    def raw_train_df(self, adapter: CMAPSSAdapter) -> pd.DataFrame:
        train_path = WORKSPACE_ROOT / "data" / "raw" / "cmapss" / "train_FD001.txt"
        assert train_path.exists(), f"Raw file missing: {train_path}"
        return adapter.load_raw(train_path)

    @pytest.fixture
    def raw_test_df(self, adapter: CMAPSSAdapter) -> pd.DataFrame:
        test_path = WORKSPACE_ROOT / "data" / "raw" / "cmapss" / "test_FD001.txt"
        assert test_path.exists(), f"Raw file missing: {test_path}"
        return adapter.load_raw(test_path)

    @pytest.fixture
    def rul_path(self) -> Path:
        p = WORKSPACE_ROOT / "data" / "raw" / "cmapss" / "RUL_FD001.txt"
        assert p.exists(), f"Raw RUL file missing: {p}"
        return p

    def test_load_raw_columns(self, raw_train_df: pd.DataFrame):
        """Validates that C-MAPSS loads exactly 26 columns per specification."""
        assert len(raw_train_df.columns) == 26
        assert "unit_number" in raw_train_df.columns
        assert "time_cycles" in raw_train_df.columns
        for i in range(1, 22):
            assert f"s_{i}" in raw_train_df.columns

    def test_cycle_monotonicity(self, adapter: CMAPSSAdapter, raw_train_df: pd.DataFrame):
        """Confirms time_cycles strictly increase by 1 per cycle with zero gaps."""
        runs = adapter.parse_runs(raw_train_df, is_train=True)
        for unit_name, run_df in runs.items():
            cycles = run_df["time_cycles"].values
            diffs = np.diff(cycles)
            assert np.all(diffs == 1), f"Cycle non-monotonicity or gap detected in {unit_name}"

    def test_train_rul_formula(self, adapter: CMAPSSAdapter, raw_train_df: pd.DataFrame):
        """Confirms training RUL formula: RUL(t) = max_cycle - t, hitting exactly 0 at failure."""
        runs = adapter.parse_runs(raw_train_df, is_train=True)
        for unit_name, run_df in runs.items():
            max_c = run_df["time_cycles"].max()
            expected_rul = (max_c - run_df["time_cycles"]).astype(float)
            actual_rul = run_df["target_rul"]
            np.testing.assert_allclose(actual_rul, expected_rul, err_msg=f"Train RUL formula mismatch in {unit_name}")
            assert actual_rul.iloc[-1] == 0.0, f"Final cycle in {unit_name} must have RUL=0"
            assert actual_rul.iloc[0] == max_c - 1, f"First cycle RUL must be max_cycle - 1 in {unit_name}"

    def test_test_rul_formula(self, adapter: CMAPSSAdapter, raw_test_df: pd.DataFrame, rul_path: Path):
        """Confirms test RUL formula: RUL(t) = RUL_final + (max_cycle - t) using verified RUL_FD001.txt."""
        runs = adapter.parse_runs(raw_test_df, is_train=False, rul_file_path=rul_path)
        ground_truth_ruls = pd.read_csv(rul_path, sep=r"\s+", header=None)[0].values

        for idx, rul_final in enumerate(ground_truth_ruls, start=1):
            unit_name = f"unit_{idx:03d}"
            assert unit_name in runs, f"Missing test run: {unit_name}"
            run_df = runs[unit_name]
            max_c = run_df["time_cycles"].max()
            actual_final_rul = run_df["target_rul"].iloc[-1]
            assert actual_final_rul == float(rul_final), (
                f"Test engine {unit_name} final RUL mismatch: expected {rul_final}, got {actual_final_rul}"
            )
            # Verify earlier cycles scale by exact delta
            expected_first_rul = float(rul_final) + (max_c - 1)
            assert run_df["target_rul"].iloc[0] == expected_first_rul


class TestAI4IAdapter:
    """Test suite for AI4I 2020 Predictive Maintenance Dataset Adapter."""

    @pytest.fixture
    def adapter(self) -> AI4IAdapter:
        return AI4IAdapter(dataset_version="1.0.0")

    @pytest.fixture
    def raw_df(self, adapter: AI4IAdapter) -> pd.DataFrame:
        csv_path = WORKSPACE_ROOT / "data" / "raw" / "ai4i2020" / "ai4i2020.csv"
        assert csv_path.exists(), f"Raw file missing: {csv_path}"
        return adapter.load_raw(csv_path)

    def test_parser_correctness(self, adapter: AI4IAdapter, raw_df: pd.DataFrame):
        """Confirms 10,000 independent rows with 10,000 unique Product IDs."""
        assert len(raw_df) == 10000
        assert raw_df["Product ID"].nunique() == 10000
        runs = adapter.parse_runs(raw_df)
        total_parsed = sum(len(df) for df in runs.values())
        assert total_parsed == 10000

    def test_udi_monotonicity(self, raw_df: pd.DataFrame):
        """Confirms UDI sequences are consecutive from 1 to 10,000."""
        udis = raw_df["UDI"].values
        diffs = np.diff(udis)
        assert np.all(diffs == 1), "AI4I UDI sequence is not strictly monotonic consecutive integers"

    def test_window_size_greater_than_one_rejection(self, adapter: AI4IAdapter, raw_df: pd.DataFrame):
        """Confirms hard rejection when window_size > 1 is requested on tabular single-snapshot data."""
        run_data = adapter.parse_runs(raw_df)
        with pytest.raises(ValueError, match="Cannot extract window_size="):
            adapter.extract_windows(run_data, window_size=20, stride=5)

        with pytest.raises(ValueError, match="violates docs/NO_HALLUCINATION_POLICY.md"):
            adapter.extract_windows(run_data, window_size=2, stride=1)

    def test_window_size_one_success(self, adapter: AI4IAdapter, raw_df: pd.DataFrame):
        """Confirms static snapshot extraction succeeds when window_size=1."""
        sample_subset = raw_df.head(10)
        run_data = adapter.parse_runs(sample_subset)
        samples = adapter.extract_windows(run_data, window_size=1)
        assert len(samples) == 10
        for s in samples:
            for mod_name, arr in s.modality_values.items():
                assert arr.shape[0] == 1, f"Expected T=1 for static observation, got shape {arr.shape}"
