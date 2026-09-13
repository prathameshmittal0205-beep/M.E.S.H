"""Unit tests for sliding window and static snapshot extraction.

Validates:
- Exact window tensor shapes: (T=20, C_m) for C-MAPSS and (T=1, C_m) for AI4I
- Strict zero cross-run bleeding: no window ever stitches observations across engine/asset boundaries
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipelines.data.adapters.cmapss import CMAPSSAdapter
from pipelines.data.adapters.ai4i import AI4IAdapter


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


class TestWindowing:
    """Test suite for sliding window extraction and run boundary isolation."""

    def test_cmapss_exact_window_shapes(self):
        """Confirms C-MAPSS FD001 windows have exactly T=20 timesteps per modality."""
        adapter = CMAPSSAdapter(subdataset="FD001", drop_invariant_sensors=False)
        raw_path = WORKSPACE_ROOT / "data" / "raw" / "cmapss" / "train_FD001.txt"
        df = adapter.load_raw(raw_path)

        # Slice 2 engines for quick test
        subset_df = df[df["unit_number"].isin([1, 2])].copy()
        runs = adapter.parse_runs(subset_df, is_train=True)
        samples = adapter.extract_windows(runs, window_size=20, stride=5)

        assert len(samples) > 0
        expected_channels = {
            "temperatures": 4,
            "pressures": 5,
            "speeds": 6,
            "gas_flow": 6,
            "operational_settings": 3,
        }

        for sample in samples:
            for mod_name, n_channels in expected_channels.items():
                assert mod_name in sample.modality_values
                arr = sample.modality_values[mod_name]
                assert arr.shape == (20, n_channels), f"Expected shape (20, {n_channels}) for {mod_name}, got {arr.shape}"

    def test_cmapss_no_cross_run_bleeding(self):
        """Confirms that no temporal window ever spans cycles across two distinct engine units."""
        adapter = CMAPSSAdapter(subdataset="FD001")
        raw_path = WORKSPACE_ROOT / "data" / "raw" / "cmapss" / "train_FD001.txt"
        df = adapter.load_raw(raw_path)

        # Test engines 1 and 2
        subset_df = df[df["unit_number"].isin([1, 2])].copy()
        runs = adapter.parse_runs(subset_df, is_train=True)
        samples = adapter.extract_windows(runs, window_size=20, stride=5)

        # Group sample IDs by engine
        engine_1_samples = [s for s in samples if s.metadata.asset_id == "unit_001"]
        engine_2_samples = [s for s in samples if s.metadata.asset_id == "unit_002"]

        assert len(engine_1_samples) > 0
        assert len(engine_2_samples) > 0

        # Verify all engine 1 samples have continuous window ranges strictly <= unit 1 max cycle
        u1_max_cycle = runs["unit_001"]["time_cycles"].max()
        for s in engine_1_samples:
            assert s.metadata.run_id == "unit_001"
            assert s.metadata.window_start >= 1
            assert s.metadata.window_end <= u1_max_cycle
            assert s.metadata.window_end - s.metadata.window_start + 1 == 20

        # Verify all engine 2 samples have continuous window ranges strictly <= unit 2 max cycle
        u2_max_cycle = runs["unit_002"]["time_cycles"].max()
        for s in engine_2_samples:
            assert s.metadata.run_id == "unit_002"
            assert s.metadata.window_start >= 1
            assert s.metadata.window_end <= u2_max_cycle
            assert s.metadata.window_end - s.metadata.window_start + 1 == 20

    def test_ai4i_exact_window_shapes(self):
        """Confirms AI4I 2020 static snapshots have exactly T=1 timestep per modality."""
        adapter = AI4IAdapter()
        raw_path = WORKSPACE_ROOT / "data" / "raw" / "ai4i2020" / "ai4i2020.csv"
        df = adapter.load_raw(raw_path).head(25)

        runs = adapter.parse_runs(df)
        samples = adapter.extract_windows(runs, window_size=1)

        assert len(samples) == 25
        expected_channels = {
            "temperature": 2,
            "speed": 1,
            "torque": 1,
            "tool_wear": 1,
        }

        for sample in samples:
            for mod_name, n_channels in expected_channels.items():
                assert mod_name in sample.modality_values
                arr = sample.modality_values[mod_name]
                assert arr.shape == (1, n_channels), f"Expected shape (1, {n_channels}) for {mod_name}, got {arr.shape}"

    def test_ai4i_no_cross_asset_stitching(self):
        """Confirms each AI4I sample represents strictly 1 distinct machine Product ID."""
        adapter = AI4IAdapter()
        raw_path = WORKSPACE_ROOT / "data" / "raw" / "ai4i2020" / "ai4i2020.csv"
        df = adapter.load_raw(raw_path).head(50)

        runs = adapter.parse_runs(df)
        samples = adapter.extract_windows(runs, window_size=1)

        product_ids = [s.metadata.asset_id for s in samples]
        assert len(set(product_ids)) == 50, "Duplicate or stitched asset IDs found in AI4I static samples"
