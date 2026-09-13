# DATASET CARD: AI4I 2020 Predictive Maintenance Dataset (Synthetic Benchmark)

Dataset card following `docs/data/DATASET_CARD_TEMPLATE.md` and verified provenance in `data/raw/provenance.json`.

## Dataset identity

- Dataset ID: `ai4i2020`
- Version: `1.0.0`
- Source: UCI Machine Learning Repository (Matzka, 2020). Source URL: `https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance` (cited in `docs/data/DATASET_STRATEGY.md` line 59).
- Download URL: `https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip`
- License/terms: Unverified in raw CSV file (UCI web catalog lists CC BY 4.0, but no license file is bundled in the download archive).
- SHA-256: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`
- Download date: `2026-09-13T20:59:51Z`

## Physical system

- Machine/equipment: Synthetic process model of a milling machine.
- Experiment/protocol: Generative model simulating 10,000 independent milling machine operating states with wear and thermal stress.
- Asset/run identifiers: `Product ID` (column 2, 10,000 unique values, 1 observation per machine) and `UDI` (column 1, row index 1 to 10,000).
- Run-to-failure available: Partial / Synthetic (single-point binary failure classifications; no multi-timestep run-to-failure trajectories).

## Measurements

| Native sensor/modality | Unit | Sampling | Meaning | Missingness |
|---|---|---|---|---|
| `Air temperature [K]` (temperature) | K (Kelvin) | Static snapshot | Ambient air temperature | None (0%) |
| `Process temperature [K]` (temperature) | K (Kelvin) | Static snapshot | Milling machine process temperature | None (0%) |
| `Rotational speed [rpm]` (speed) | rpm | Static snapshot | Spindle rotational speed calculated from 2860 W power with normal noise | None (0%) |
| `Torque [Nm]` (torque) | Nm | Static snapshot | Torque around 40 Nm with normal noise (no negative values) | None (0%) |
| `Tool wear [min]` (tool_wear) | min | Static snapshot | Tool wear duration in minutes (quality variants H/M/L add 5/3/2 min) | None (0%) |

## Targets

| Target | Present in source? | Definition | Valid task |
|---|---|---|---|
| RUL | No | Continuous RUL is NOT available (1 observation per machine) | Invalid for AI4I |
| `Machine failure` | Yes (column 9) | Binary indicator (1 if any failure mode triggered, 0 otherwise; 339/10,000 positive, 3.39%) | Binary failure classification |
| `TWF` (Tool Wear Failure) | Yes (column 10) | Tool wear failure flag (wear between 200 and 240 mins; 46 occurrences) | Multi-label / fault mode classification |
| `HDF` (Heat Dissipation Failure) | Yes (column 11) | Heat dissipation failure flag ($\Delta T < 8.6\text{ K}$ and speed $< 1380\text{ rpm}$; 115 occurrences) | Multi-label / fault mode classification |
| `PWF` (Power Failure) | Yes (column 12) | Power failure flag (Power $< 3500\text{ W}$ or $> 9000\text{ W}$; 95 occurrences) | Multi-label / fault mode classification |
| `OSF` (Overstrain Failure) | Yes (column 13) | Overstrain failure flag (product of tool wear and torque exceeds threshold; 98 occurrences) | Multi-label / fault mode classification |
| `RNF` (Random Failure) | Yes (column 14) | Random failure flag (0.1% independent random process failure; 19 occurrences) | Multi-label / fault mode classification |

## Split

- Split strategy: Stratified asset-level partitioning (`asset_stratified_70_15_15`). Partitions distinct machines by unique `Product ID` while stratifying on `Machine failure` to preserve rare failure rate ($3.39\%$).
- Train runs/assets: 7,000 distinct machine assets (237 positive failures, $3.39\%$).
- Validation runs/assets: 1,500 distinct machine assets (51 positive failures, $3.40\%$).
- Test runs/assets: 1,500 distinct machine assets (51 positive failures, $3.40\%$).
- Leakage checks: `verify_no_overlap()` mathematically confirms zero `Product ID` overlap between train, val, and test subsets (`leakage_verified: true` in `ai4i2020_split.json`).

## Processing

- Window length: $T=1$ (static snapshot per machine observation). Sliding windows across distinct rows ($T > 1$) are strictly forbidden to prevent fabricating artificial temporal continuity across unrelated machines per `docs/NO_HALLUCINATION_POLICY.md`.
- Stride: None ($T=1$).
- Normalization: Per-modality z-score standard scaling fit strictly on the 7,000 training assets. Frozen parameters $(\mu, \sigma)$ applied to val, test, and ablation splits without refitting.
- Feature derivation: Grouped into 4 physical modalities: `temperature` (2 ch), `speed` (1 ch), `torque` (1 ch), `tool_wear` (1 ch). Static feature: `Type` encoded (L=0, M=1, H=2).
- Preprocessor version: `v1.0_static_snapshot`.

## Missing modality

- Natural missingness: None (0.0%). All columns populated across all 10,000 rows.
- Controlled dropout procedure: Synthetic test-time ablation suites generated strictly from the held-out test split (1,500 samples $\times$ 4 modalities = 6,000 ablation samples).
- Mask definition: `modality_mask[m] = 0` denotes a dropped modality; values are zeroed out (`0.0`) for tensor regularity without leakage. Stamped with `missingness_type="synthetic_controlled_dropout"`.

## Limitations

1. **Synthetic Data**: Explicitly documented as synthetic in MESH docs (`README.md` Section 5, `docs/data/DATASET_STRATEGY.md` line 17, `docs/MESH_DATASET_DECISION_MATRIX.md` line 11).
2. **Tabular, Non-Temporal**: Each row is an independent observation of a different machine. It contains NO multi-cycle time-series trajectories.
3. **Restricted Scope**: In-scope strictly for data loader verification, pipeline sanity testing, and classical baseline comparisons. Must NEVER be claimed as evidence of real industrial machine health monitoring.

## Approval

- Reviewed by: Naman (Data Engineering Owner, MESH)
- Date: 2026-09-13
- Approved for which experiments: Data loader testing, API sanity validation, baseline binary failure classification. Excluded from primary prognostics claims.
