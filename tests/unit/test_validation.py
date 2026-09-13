"""Unit tests for data quality validation module.

Validates:
- QA gate correctly fails on injected corrupt records (NaN/Inf, duplicates, cycle breaks, negative physical readings)
- Empirical bounds & engineering estimates never gate passed_qa (strictly informational)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipelines.data.adapters.cmapss import CMAPSSAdapter
from pipelines.data.adapters.ai4i import AI4IAdapter
from pipelines.data.validation import DataQualityValidator


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


class TestValidationQAGate:
    """Test suite for DataQualityValidator QA gates and corruption handling."""

    @pytest.fixture
    def clean_cmapss_df(self) -> pd.DataFrame:
        adapter = CMAPSSAdapter(subdataset="FD001")
        raw_path = WORKSPACE_ROOT / "data" / "raw" / "cmapss" / "train_FD001.txt"
        return adapter.load_raw(raw_path)

    @pytest.fixture
    def clean_ai4i_df(self) -> pd.DataFrame:
        adapter = AI4IAdapter()
        raw_path = WORKSPACE_ROOT / "data" / "raw" / "ai4i2020" / "ai4i2020.csv"
        return adapter.load_raw(raw_path)

    def test_clean_data_passes_qa(self, clean_cmapss_df: pd.DataFrame, clean_ai4i_df: pd.DataFrame):
        """Confirms pristine raw datasets pass the hard structural QA gate."""
        val_cmapss = DataQualityValidator(dataset_id="nasa_cmapss_fd001")
        rep_c, _ = val_cmapss.validate_cmapss(clean_cmapss_df, is_train=True)
        assert rep_c.passed_qa is True
        assert rep_c.missing_value_count == 0
        assert rep_c.duplicate_records_count == 0
        assert rep_c.monotonicity_violations == 0

        val_ai4i = DataQualityValidator(dataset_id="ai4i2020")
        rep_a, _ = val_ai4i.validate_ai4i(clean_ai4i_df)
        assert rep_a.passed_qa is True
        assert rep_a.missing_value_count == 0
        assert rep_a.duplicate_records_count == 0
        assert rep_a.monotonicity_violations == 0

    def test_injected_missing_nan_fails_qa(self, clean_cmapss_df: pd.DataFrame):
        """Confirms injected NaN value trips the hard QA gate."""
        corrupt_df = clean_cmapss_df.copy()
        corrupt_df.loc[10, "s_2"] = np.nan

        val = DataQualityValidator(dataset_id="nasa_cmapss_fd001")
        report, _ = val.validate_cmapss(corrupt_df)
        assert report.passed_qa is False
        assert report.missing_value_count >= 1

    @pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
    def test_injected_infinite_value_fails_qa(self, clean_ai4i_df: pd.DataFrame):
        """Confirms injected Inf value trips the hard QA gate."""
        corrupt_df = clean_ai4i_df.copy()
        # Air temperature [K] is float64; inject np.inf
        corrupt_df.loc[5, "Air temperature [K]"] = np.inf

        val = DataQualityValidator(dataset_id="ai4i2020")
        report, _ = val.validate_ai4i(corrupt_df)
        assert report.passed_qa is False
        assert report.missing_value_count >= 1

    def test_injected_duplicate_fails_qa(self, clean_cmapss_df: pd.DataFrame):
        """Confirms duplicate (unit_number, time_cycles) trips the hard QA gate."""
        dup_row = clean_cmapss_df.iloc[[0]].copy()
        corrupt_df = pd.concat([clean_cmapss_df, dup_row], ignore_index=True)

        val = DataQualityValidator(dataset_id="nasa_cmapss_fd001")
        report, _ = val.validate_cmapss(corrupt_df)
        assert report.passed_qa is False
        assert report.duplicate_records_count >= 1

    def test_injected_monotonicity_gap_fails_qa(self, clean_cmapss_df: pd.DataFrame):
        """Confirms non-monotonic cycle sequence trips the hard QA gate."""
        corrupt_df = clean_cmapss_df.copy()
        # Invert cycles 5 and 6 in unit 1
        corrupt_df.loc[4, "time_cycles"] = 10
        corrupt_df.loc[5, "time_cycles"] = 5

        val = DataQualityValidator(dataset_id="nasa_cmapss_fd001")
        report, _ = val.validate_cmapss(corrupt_df)
        assert report.passed_qa is False
        assert report.monotonicity_violations >= 1

    def test_injected_negative_sensor_fails_qa(self, clean_ai4i_df: pd.DataFrame):
        """Confirms negative Kelvin temperature trips the physical non-negativity gate."""
        corrupt_df = clean_ai4i_df.copy()
        corrupt_df.loc[12, "Air temperature [K]"] = -5.0

        val = DataQualityValidator(dataset_id="ai4i2020")
        report, _ = val.validate_ai4i(corrupt_df)
        assert report.passed_qa is False
        assert report.invalid_sensor_readings >= 1

    def test_engineering_estimates_do_not_gate_passed_qa(self, clean_cmapss_df: pd.DataFrame):
        """Confirms that unverified engineering estimate boundary alerts DO NOT fail passed_qa."""
        corrupt_df = clean_cmapss_df.copy()
        # s_2 has estimate (500.0, 800.0). Inject 950.0 (positive, non-negative, but > 800.0)
        corrupt_df.loc[0, "s_2"] = 950.0

        val = DataQualityValidator(dataset_id="nasa_cmapss_fd001")
        report, details = val.validate_cmapss(corrupt_df)

        # Confirm that out-of-estimate readings are flagged under unverified bounds
        assert "s_2" in details["engineering_estimate_out_of_bounds"]
        assert details["engineering_estimate_out_of_bounds"]["s_2"] >= 1
        assert details["bounds_source"] == "engineering_estimate_unverified"

        # Hard QA gate must still PASS because structural rules and non-negativity were satisfied
        assert report.passed_qa is True
        assert details["passed_qa"] is True
