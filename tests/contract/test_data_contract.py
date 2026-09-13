"""Data contract validation tests.

Validates final NumPy .npz artifacts against the integration schema contract:
- Array key naming convention: modality_{name}, mask_{name}, target_{name}, sample_ids
- Correct array dimensionality: modality is (N, T, C_m), mask is (N,), target is (N,)
- Strict dtypes: float32 for modalities, int32 for masks, unicode string for sample_ids
- Strict portability: file loads with np.load(..., allow_pickle=False)
- Manifest JSON mirrors array shapes and statistics accurately
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


class TestDataContract:
    """Contract verification for ML data ingestion deliverables."""

    @pytest.mark.parametrize("dataset_folder", ["cmapss_fd001", "ai4i2020"])
    @pytest.mark.parametrize("split_name", ["train", "val", "test"])
    def test_canonical_npz_contract(self, dataset_folder: str, split_name: str):
        """Validates that all train, val, and test .npz files strictly fulfill the integration contract."""
        split_dir = WORKSPACE_ROOT / "data" / "processed" / dataset_folder / split_name
        npz_path = split_dir / "data.npz"
        manifest_path = split_dir / "manifest.json"

        assert npz_path.exists(), f"Deliverable data.npz missing at {npz_path}"
        assert manifest_path.exists(), f"Deliverable manifest.json missing at {manifest_path}"

        # 1. Load without pickle (must be purely portable binary arrays)
        with np.load(npz_path, allow_pickle=False) as npz:
            keys = list(npz.files)

            # 2. Must contain sample_ids array
            assert "sample_ids" in keys
            sample_ids = npz["sample_ids"]
            n_samples = len(sample_ids)
            assert n_samples > 0
            assert sample_ids.dtype.kind in ("U", "S"), f"sample_ids must be string/unicode, got {sample_ids.dtype}"

            # 3. Verify modality tensors and binary masks
            modality_keys = [k for k in keys if k.startswith("modality_")]
            assert len(modality_keys) > 0, "No modality arrays found in .npz"

            for mod_k in modality_keys:
                mod_name = mod_k.replace("modality_", "")
                mask_k = f"mask_{mod_name}"
                assert mask_k in keys, f"Missing corresponding mask array for {mod_k}"

                mod_arr = npz[mod_k]
                mask_arr = npz[mask_k]

                # Modality tensor contract: 3D array of shape (N, T, C_m)
                assert mod_arr.ndim == 3, f"Expected 3D tensor (N, T, C_m), got ndim={mod_arr.ndim} for {mod_k}"
                assert mod_arr.shape[0] == n_samples, f"Batch dim {mod_arr.shape[0]} does not match N={n_samples}"
                assert mod_arr.dtype == np.float32, f"Expected float32 for {mod_k}, got {mod_arr.dtype}"
                assert not np.isnan(mod_arr).any(), f"NaN values detected in {mod_k}"
                assert not np.isinf(mod_arr).any(), f"Inf values detected in {mod_k}"

                # Mask contract: 1D array of shape (N,) with int32 dtype
                assert mask_arr.ndim == 1, f"Expected 1D mask (N,), got ndim={mask_arr.ndim} for {mask_k}"
                assert mask_arr.shape[0] == n_samples, f"Mask dim {mask_arr.shape[0]} does not match N={n_samples}"
                assert mask_arr.dtype == np.int32, f"Expected int32 mask, got {mask_arr.dtype}"
                assert np.all(np.isin(mask_arr, [0, 1])), f"Mask values must be binary (0 or 1) in {mask_k}"

            # 4. Verify target arrays
            target_keys = [k for k in keys if k.startswith("target_")]
            assert len(target_keys) > 0, "No target arrays found in .npz"
            for tgt_k in target_keys:
                tgt_arr = npz[tgt_k]
                assert tgt_arr.ndim == 1, f"Expected 1D target (N,), got ndim={tgt_arr.ndim} for {tgt_k}"
                assert tgt_arr.shape[0] == n_samples, f"Target dim {tgt_arr.shape[0]} does not match N={n_samples}"

        # 5. Cross-verify with JSON manifest
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["sample_count"] == n_samples
        assert manifest["split_name"] == split_name
        assert "target_statistics" in manifest
        assert "modalities" in manifest
