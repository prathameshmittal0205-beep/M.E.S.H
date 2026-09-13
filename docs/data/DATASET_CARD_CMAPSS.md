# DATASET CARD: NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset

Dataset card following `docs/data/DATASET_CARD_TEMPLATE.md` and verified provenance in `data/raw/provenance.json`.

## Dataset identity

- Dataset ID: `nasa_cmapss_fd001` (parent: `nasa_cmapss`)
- Version: `1.0.0`
- Source: C-MAPSS simulation model developed at NASA Glenn Research Center (Saxena et al., 2008, PHM08 Challenge benchmark). Documented in `data/raw/cmapss/readme.txt` and `data/external/Damage Propagation Modeling.pdf`. Source URL: `http://ti.arc.nasa.gov/projects/data_prognostics` (Paper Ref [1]); cataloged at `https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq`.
- Download URL: `https://phm-datasets.s3.amazonaws.com/NASA/CMAPSSData.zip`
- License/terms: Unverified in raw text files. Cited as public research benchmark data created under the NASA Integrated Vehicle Health Management (IVHM) program (Saxena et al., 2008).
- SHA-256:
  - `train_FD001.txt`: `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`
  - `test_FD001.txt`: `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851`
  - `RUL_FD001.txt`: `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca`
  - `readme.txt`: `4f5270554b775c67e73aff383c5436fd329d6e4cc3d3a116913276fae511269b`
- Download date: `2026-09-13T21:00:51Z`

## Physical system

- Machine/equipment: Commercial Modular Aero-Propulsion System Simulation (90,000 lb thrust class dual-spool turbofan aircraft engine model).
- Experiment/protocol: Run-to-failure degradation trajectory simulation starting from initial normal health, experiencing high-pressure compressor (HPC) degradation under sea-level cruise operational conditions (subdataset FD001).
- Asset/run identifiers: `unit_number` (identifies individual engine asset in fleet, column 1 in raw text files, mapped to `unit_001` through `unit_100`).
- Run-to-failure available: Yes (Train set runs to failure threshold; test set truncated prior to failure with ground-truth final RUL provided).

## Measurements

| Native sensor/modality | Unit | Sampling | Meaning | Missingness |
|---|---|---|---|---|
| `setting_1`, `setting_2`, `setting_3` (operational_settings) | altitude, Mach, TRA | Discrete flight cycles (1 snapshot/cycle at cruise) | Operational flight conditions (constant sea-level condition in FD001) | None (0%) |
| `s_1` (T2) (temperatures) | °R (Rankine) | Discrete flight cycles | Total temperature at fan inlet (invariant sea-level sensor) | None (0%) |
| `s_2` (T24) (temperatures) | °R (Rankine) | Discrete flight cycles | Total temperature at LPC outlet | None (0%) |
| `s_3` (T30) (temperatures) | °R (Rankine) | Discrete flight cycles | Total temperature at HPC outlet | None (0%) |
| `s_4` (T50) (temperatures) | °R (Rankine) | Discrete flight cycles | Total temperature at LPT outlet | None (0%) |
| `s_5` (P2) (pressures) | psia | Discrete flight cycles | Pressure at fan inlet (invariant sea-level sensor) | None (0%) |
| `s_6` (P15) (pressures) | psia | Discrete flight cycles | Total pressure in bypass-duct | None (0%) |
| `s_7` (P30) (pressures) | psia | Discrete flight cycles | Total pressure at HPC outlet | None (0%) |
| `s_10` (epr) (pressures) | dimensionless | Discrete flight cycles | Engine pressure ratio P50/P2 (invariant sea-level sensor) | None (0%) |
| `s_11` (Ps30) (pressures) | psia | Discrete flight cycles | Static pressure at HPC outlet | None (0%) |
| `s_8` (Nf) (speeds) | rpm | Discrete flight cycles | Physical fan speed | None (0%) |
| `s_9` (Nc) (speeds) | rpm | Discrete flight cycles | Physical core speed | None (0%) |
| `s_13` (NRf) (speeds) | rpm | Discrete flight cycles | Corrected fan speed | None (0%) |
| `s_14` (NRc) (speeds) | rpm | Discrete flight cycles | Corrected core speed | None (0%) |
| `s_18` (Nf_dmd) (speeds) | rpm | Discrete flight cycles | Demanded fan speed (invariant sea-level sensor) | None (0%) |
| `s_19` (PCNfR_dmd) (speeds) | rpm | Discrete flight cycles | Demanded corrected fan speed (invariant sea-level sensor) | None (0%) |
| `s_12` (phi) (gas_flow) | pps/psi | Discrete flight cycles | Ratio of fuel flow to Ps30 | None (0%) |
| `s_15` (BPR) (gas_flow) | dimensionless | Discrete flight cycles | Bypass Ratio | None (0%) |
| `s_16` (farB) (gas_flow) | dimensionless | Discrete flight cycles | Burner fuel-air ratio (invariant sea-level sensor) | None (0%) |
| `s_17` (htBleed) (gas_flow) | dimensionless | Discrete flight cycles | Bleed Enthalpy | None (0%) |
| `s_20` (W31) (gas_flow) | lbm/s | Discrete flight cycles | HPT coolant bleed | None (0%) |
| `s_21` (W32) (gas_flow) | lbm/s | Discrete flight cycles | LPT coolant bleed | None (0%) |

## Targets

| Target | Present in source? | Definition | Valid task |
|---|---|---|---|
| RUL (`target_rul`) | Yes (derived for train: $t_{\text{max}} - t$; from `RUL_FD001.txt` for test: $\text{RUL}_{\text{final}} + t_{\text{max}} - t$) | Remaining Useful Life in discrete operational flight cycles until maintenance/failure threshold | Remaining Useful Life regression |
| Fault class | Partial (implicit) | HPC degradation mode (single failure mode in FD001) | Single-fault baseline |
| Degradation/wear | Implicit in RUL | Monotonic thermo-dynamical wear trajectory | Continuous health index estimation |

## Split

- Split strategy: Asset-level engine unit holdout (`asset_level_engine_unit_holdout_80_20`). All time cycles and temporal windows belonging to an engine unit remain strictly inside that engine's assigned partition.
- Train runs/assets: 80 engine units (3,040 windows of length $T=20, \text{stride}=5$).
- Validation runs/assets: 20 engine units (746 windows of length $T=20, \text{stride}=5$).
- Test runs/assets: 100 official test engines from `test_FD001.txt` (2,279 windows of length $T=20, \text{stride}=5$).
- Leakage checks: `verify_no_overlap()` mathematically confirms zero engine unit overlap across train, val, and test partitions (`leakage_verified: true` in `cmapss_fd001_split.json`).

## Processing

- Window length: $T=20$ discrete flight cycles.
- Stride: $\text{stride}=5$ (configurable in `configs/data/cmapss_fd001.yaml`).
- Normalization: Per-modality z-score standard scaling fit strictly on the 80 training engine units. Scaler parameters $(\mu, \sigma)$ frozen and applied to val, test, and ablation splits without refitting. Constant/invariant channels pinned to $\mu = \text{mean}, \sigma = 1.0$, normalizing identically to $0.0$.
- Feature derivation: Grouped into 5 native physical modalities: `temperatures` (4 ch), `pressures` (5 ch), `speeds` (6 ch), `gas_flow` (6 ch), `operational_settings` (3 ch).
- Preprocessor version: `v1.0_window20_stride5`.

## Missing modality

- Natural missingness: None (0.0%). All 21 sensor columns and 3 settings populated across all 20,631 rows.
- Controlled dropout procedure: Synthetic test-time ablation suites generated strictly from the held-out test split (2,279 samples $\times$ 5 modalities = 11,395 ablation samples).
- Mask definition: `modality_mask[m] = 0` denotes a dropped modality; values are zeroed out (`0.0`) for tensor regularity without leaking unmasked numbers. Stamped with `missingness_type="synthetic_controlled_dropout"`.

## Limitations

1. **Simulation Model Output**: C-MAPSS data is generated from a numerical thermodynamic simulation code (C-MAPSS), not physical flight recorder telemetry (Damage Propagation Modeling.pdf Section III-A). Model predictions cannot be claimed as real flight evidence without physical validation.
2. **Invariant Channels**: Subdataset FD001 simulates sea-level static flight conditions, causing 6 sensors (`s_1`, `s_5`, `s_10`, `s_16`, `s_18`, `s_19`) to have zero variance across all cycles.
3. **Single Operating Condition**: FD001 contains only 1 operating regime (sea level) and 1 failure mode (HPC degradation). Multi-condition transfer requires FD002/FD004.

## Approval

- Reviewed by: Naman (Data Engineering Owner, MESH)
- Date: 2026-09-13
- Approved for which experiments: Primary Prognostics Baseline, Remaining Useful Life (RUL) regression, and multi-modal cross-attention test-time ablation experiments.
