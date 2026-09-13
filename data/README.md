# MESH Data Pipeline & Processed Artifacts

This directory contains raw source datasets, intermediate verification reports, and final canonical multi-modal datasets for MESH (*Multisensor Engine for System Health*).

Governed by `docs/data/DATA_PIPELINE.md`, `docs/data/DATASET_STRATEGY.md`, `docs/NO_HALLUCINATION_POLICY.md`, and `docs/model/INTEGRATION_CONTRACT.md`.

---

## Directory Layout

```
data/
├── raw/                                      # Pristine, read-only acquired datasets
│   ├── cmapss/                               # NASA C-MAPSS Turbofan Degradation (13 files: train, test, RUL for FD001-FD004)
│   ├── ai4i2020/                             # AI4I 2020 Predictive Maintenance (10,000 synthetic milling machine records)
│   └── provenance.json                       # SHA-256 hashes, source URLs, licensing, and metadata catalog
├── external/                                 # Official reference documentation
│   └── Damage Propagation Modeling.pdf      # NASA C-MAPSS reference paper (Saxena et al., 2008)
├── interim/                                  # Intermediate validation & QA audit artifacts
│   └── qa_reports/
│       ├── cmapss_fd001_qa.json              # Structural QA audit report & empirical bounds for C-MAPSS FD001
│       └── ai4i2020_qa.json                  # Structural QA audit report & empirical bounds for AI4I 2020
└── processed/                                # Final canonical datasets consumed by ML pipeline
    ├── splits/
    │   ├── cmapss_fd001_split.json           # Asset-level leak-free split (80 train, 20 val, 100 test engines)
    │   └── ai4i2020_split.json               # Asset-level stratified split (7,000 train, 1,500 val, 1,500 test assets)
    ├── scalers/
    │   ├── cmapss_fd001_scaler.json / .pkl   # Train-only fitted z-score normalization parameters
    │   └── ai4i2020_scaler.json / .pkl       # Train-only fitted z-score normalization parameters
    ├── cmapss_fd001/                         # C-MAPSS FD001 canonical deliverables (T=20, stride=5)
    │   ├── train/                            # 3,040 samples across 80 engines (data.npz, samples.pkl, manifest.json)
    │   ├── val/                              # 746 samples across 20 engines (data.npz, samples.pkl, manifest.json)
    │   ├── test/                             # 2,279 samples across 100 engines (data.npz, samples.pkl, manifest.json)
    │   └── ablations/                        # Test-time single-modality controlled dropout suites (2,279 samples each)
    │       ├── drop_temperatures/
    │       ├── drop_pressures/
    │       ├── drop_speeds/
    │       ├── drop_gas_flow/
    │       └── drop_operational_settings/
    └── ai4i2020/                             # AI4I 2020 canonical deliverables (T=1 static snapshots)
        ├── train/                            # 7,000 samples across 7,000 assets (data.npz, samples.pkl, manifest.json)
        ├── val/                              # 1,500 samples across 1,500 assets (data.npz, samples.pkl, manifest.json)
        ├── test/                             # 1,500 samples across 1,500 assets (data.npz, samples.pkl, manifest.json)
        └── ablations/                        # Test-time single-modality controlled dropout suites (1,500 samples each)
            ├── drop_temperature/
            ├── drop_speed/
            ├── drop_torque/
            └── drop_tool_wear/
```

---

## Canonical Data Artifact Format (`data.npz`)

The primary deliverable for model training and evaluation is the portable compressed NumPy archive (`data.npz`), loadable with `np.load(..., allow_pickle=False)`:

- **Modality Tensors** (`modality_{name}`): `float32` array of shape `(N, T, C_m)`:
  - `N`: Number of canonical samples in the partition.
  - `T`: Sequence length ($T=20$ for C-MAPSS FD001; $T=1$ for AI4I 2020).
  - `C_m`: Number of physical channels in modality $m$.
- **Binary Modality Masks** (`mask_{name}`): `int32` array of shape `(N,)` where `1` = available, `0` = dropped.
- **Targets** (`target_{name}`):
  - `target_rul`: `float32` array of shape `(N,)` (C-MAPSS Remaining Useful Life).
  - `target_binary_failure`: `int32` array of shape `(N,)` (AI4I Machine failure flag, 0 or 1).
  - `target_fault_class`: `int32` array of shape `(N,)` (AI4I multi-class failure category, 0-5).
- **Sample Identifiers** (`sample_ids`): Unicode string array of shape `(N,)` uniquely identifying each window.

Each partition also includes `manifest.json` recording sample counts, run lists, channel distributions, and target statistics.

---

## Reproducing the Pipeline from Scratch

To clean and regenerate all processed datasets end-to-end:

### 1. Execute C-MAPSS FD001 Pipeline
```bash
python -m pipelines.data.pipeline --config configs/data/cmapss_fd001.yaml --clean
```

### 2. Execute AI4I 2020 Pipeline
```bash
python -m pipelines.data.pipeline --config configs/data/ai4i2020.yaml --clean
```

### 3. Run Automated Verification Test Suite
```bash
python -m pytest -v
```

All parameters (window lengths, strides, split ratios, random seeds, invariant sensor behaviors, and target columns) are configured in `configs/data/cmapss_fd001.yaml` and `configs/data/ai4i2020.yaml`. Zero magic numbers remain in source code.
