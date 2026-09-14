# MESH Project: Final Technical Report

## 1. Documentation & Data Contract
* **Documentation Read:** `docs/model/MODEL_SPEC.md`, `docs/data/DATA_DICTIONARY.md`, `docs/architecture/SYSTEM_ARCHITECTURE.md`, `docs/training/TRAINING_EVALUATION.md`, `docs/training/MISSING_MODALITY_EXPERIMENTS.md`, `docs/explainability/EXPLAINABILITY.md`, `docs/data/DATA_PIPELINE.md`, `docs/model/UNCERTAINTY_CALIBRATION.md`, `docs/team/PRATHAMESH_MODEL.md`, `docs/team/INTEGRATION_CONTRACT.md`, `docs/DEFINITION_OF_DONE.md`, `docs/data/DATASET_STRATEGY.md`, `docs/data/DATA_PROVENANCE.md`, and `docs/NO_HALLUCINATION_POLICY.md`.
* **Data Contract Consumed:** Implemented and validated `CanonicalBatch` according to the exact Pydantic spec (`ml/data/contract.py`), allowing arbitrary tensors and strictly typing all metadata.

## 2. Architecture Implemented
Implemented the `MESHModel` (`ml/models/mesh_model.py`) natively in PyTorch:
* **Modality Encoders:** 1D CNNs (kernel=3, out_channels=8) into Bidirectional LSTMs (hidden=16) per modality (`temperature`, `tool_wear`, `rotational_speed`, `torque`).
* **Fusion:** Mask-Aware Cross-Attention (`ml/models/fusion.py`) using PyTorch's `MultiheadAttention`. Missing modalities are explicitly zeroed out of the attention softmax (both queries and keys) using `-inf` masking.
* **Temporal Refiner:** Standard TransformerEncoder layer processing the fused sequence.
* **Heads:** 
  * RUL (Regression): Outputs Mean and Log-Variance (clamped to `[-10, 10]`) for NLL optimization.
  * Fault (Classification): 3-class logits.
  * Anomaly (Binary): Single logit for sigmoid activation.

## 3. Files Created / Modified
* `ml/models/mesh_model.py`, `ml/models/encoders.py`, `ml/models/fusion.py`, `ml/models/temporal.py`, `ml/models/heads.py`
* `pipelines/training/train.py`, `pipelines/training/tracker.py`
* `pipelines/evaluation/evaluate.py`, `pipelines/evaluation/missing_modality_experiment.py`
* `ml/explainability/attribution.py`
* `ml/inference/interface.py`
* `tests/ml/test_model_architecture.py`, `tests/ml/test_contract.py`

## 4. Training & Evaluation Pipeline
* **Training (`train.py`):** Trains using NLL (RUL) + CrossEntropy (Fault) + BCE (Anomaly). Integrates a JSONL experiment tracker. Includes deterministic 80/20 train/val splits. Persists checkpoint metadata (model version, config).
* **Evaluation (`evaluate.py`):** Computes exact MAE, RMSE, NLL, and standard classification metrics. RUL metrics are correctly de-normalized into real cycle units.

## 5. Missing-Modality Experiments
Executed an 8-condition systematic dropout sweep (0 to 4 modalities dropped) via `missing_modality_experiment.py`. Output is saved as structured JSON artifacts. 

## 6. Uncertainty, Calibration, and Explainability
* **Explainability (`attribution.py`):** Extracted native attention weights (proving exactly $0.0$ attention for masked modalities). Used Captum GradientSHAP for feature attribution.
* **Uncertainty:** RUL head successfully learned aleatoric variance (preventing variance collapse via soft clamping).

## 7. Checkpoint Location
The best trained model checkpoint is persisted at:
`checkpoints/model_best.pt`

## 8. Inference Contract
Implemented a strict, decoupled Python inference engine (`ml/inference/interface.py`).
* Consumes `CanonicalBatch`, produces `PredictionBundle`.
* Transparently handles denormalization to real units (cycles).
* Embeds traceability metadata (`model_version` timestamp, `checkpoint_id`, `training_config_ref`).
* Exposes `attention_weights` strictly as an opt-in parameter for payload efficiency.

## 9. Testing
Built strict PyTest suites (`tests/ml/`) that are **100% Passing**:
* **`test_model_architecture.py`**: Verifies exact tensor shapes, masked-modality zeroing, NaN-safety under total dropout, and deterministic eval bounds.
* **`test_contract.py`**: Verifies `CanonicalBatch` rejection of malformed data, `PredictionBundle` JSON serialization, and full checkpoint save/load equivalence.

## 10. Assumptions & Known Limitations
> [!WARNING]
> **Mock Data Performance:** All metrics, attributions, and variance boundaries currently reflect convergence on uniform noise. The model currently exhibits majority-class collapse (expected) and tiny/random feature attributions. Real predictive evaluation is pending actual data.

> [!WARNING]
> **AI4I Class Imbalance & Evaluation:** The initial "perfect" AI4I RUL metrics were exposed as a leak, and the model was refactored for its true task: Binary Failure and Fault Type Classification. While the new model achieved 97.47% accuracy on anomaly detection, the test set has a severe class imbalance (96.6% normal, 3.4% failures). A true evaluation of the positive (failure) class yields: **Precision: 0.7826, Recall: 0.3529, F1: 0.4865**. The model misses ~65% of true failures, though it is highly confident when it does flag one. Fault classification struggles similarly (Macro-F1: 0.2703). This is an honest baseline for 3 epochs on a highly imbalanced dataset, and accuracy alone should not be used as the headline metric. **Future tuning note:** The low recall is a classic symptom of training on highly imbalanced data with unweighted BCE; whoever tunes this model next should consider using `pos_weight` in `BCEWithLogitsLoss` or a focal loss to penalize minority-class misses more heavily.

> [!WARNING]
> **Epistemic Uncertainty limitation:** Our missing-modality experiment empirically proved that NLL (Aleatoric) variance remains completely flat (~3060) when input modalities drop out. The loss formulation measures target noise, not model confidence. An Epistemic method (like MC Dropout) MUST be implemented before deployment to flag "I don't know" when sensors fail. **Update:** MC Dropout as an epistemic method is now supported by an actual modality-dropout-trained model (implemented as a p=0.15 regularizer during training), rather than being justified by a mechanism that didn't exist yet.

> [!WARNING]
> **Hardcoded Normalization:** RUL target normalization stats (mean, std) are currently hardcoded placeholders injected during training initialization. These must be replaced with the actual dataset statistics computed by Naman's preprocessor.

## 11. Backend Integration (For Sarthak)
* The `PredictionBundle` is fully JSON-serializable via `dataclasses.asdict()`.
* All regression metrics are returned in absolute real units (no z-score decoding required on the backend).
* `attention_weights` are natively suppressed to save bandwidth. Pass `return_attention=True` to `engine.predict()` if visual explainability is requested by the dashboard.
* Traceability is guaranteed: parse `model_version` directly from the `PredictionBundle` to log exactly which checkpoint generated the output.

## 12. CMAPSS vs. AI4I Architectural Split
> [!NOTE]
> **Design Decision:** The architecture is formally split into two native model configurations due to strictly disjoint modalities and different temporal assumptions between the target datasets.

* **CMAPSS (`cmapss_model.yaml`):** A 24-channel temporal problem ($T=20$) utilizing 5 logical modalities (`temperatures`, `pressures`, `speeds`, `gas_flow`, `operational_settings`). It uses the full `MESHModel` (CNN + BiLSTM + Temporal Transformer). Because CMAPSS provides no classification targets, the Fault and Anomaly heads are strictly bypassed during training and inference.
* **AI4I (`ai4i_model.yaml`):** A 4-channel static problem ($T=1$). The temporal layers were identified as wasteful and physically incorrect for this dataset. We implemented a new lightweight `StaticModalityEncoder` (MLP) mapping static snapshots to embeddings. It reuses the mask-aware cross-attention fusion seamlessly without a temporal dimension and directly predicts all 3 heads (RUL, Fault, Anomaly).

**Current Status (Blocked):** We are currently awaiting Naman to provide a mock data fixture (`tests/fixtures/`) that represents the real, compiled data schemas to begin real-data training. The ML inference and validation layer is fully prepared to consume this data the moment it drops.

## 13. Open Issues (For Naman)
> [!IMPORTANT]
> **Dataset Selection & Modality Alignment:** Per the project README, AI4I 2020 is a synthetic benchmark only and cannot be used for our real-data claims. The real candidate datasets are NASA FEMTO/PRONOSTIA, NASA IMS Bearings, NASA Milling Wear, and QIT-CEMC. Because no single real dataset perfectly provides the original DA1 four channels (temperature, rotational speed, torque, tool wear) alongside a clean RUL target, Naman must finalize the real dataset selection and output its **native** channels. We will not fabricate a 4-channel dataset. Once the real dataset is chosen, the model's `native_modalities` configuration must be updated to align with the actual sensors provided by that dataset.

## 13. Quality Assurance: Bugs Caught & Resolved
The following concrete issues were caught and fixed during the interactive modeling phase, documented here as a historical record for the team:
* **RUL NLL Explosion:** Target RUL was left unnormalized, forcing the NLL loss to NaN; resolved via target z-scoring and log-variance soft-clamping.
* **Fusion Mask Leakage:** The attention layer leaked the masked modality's own query vector into the output; fixed by multiplying the MHA output by the binary mask before pooling.
* **Total-Dropout NaNs:** Dropping 100% of sensors produced 0/0 NaN division in the fusion pooling; fixed via `nan_to_num` and safe denominator handling.
* **Traceability Gap:** `model_version` was hardcoded as "unknown" in the inference payload; fixed by baking a real version string directly into the checkpoint at save time.
* **Explainability Payload Bloat:** `attention_weights` were silently stripped from the inference bundle; fixed via an explicit `return_attention` opt-in parameter.
* **Silent Test Passes:** `test_masked_modality_zeroing` had a conditional `hasattr` guard that could silently skip the test; replaced with a hard assertion.
* **Checkpoint Test Fallacy:** `test_checkpoint_roundtrip` originally tested `load -> load` agreement rather than `save -> load` fidelity; rewriting to a true save-and-reload cycle caught two real downstream schema and variance-conversion bugs.
* **Modality Naming Mismatch:** Modality naming mismatch (vibration vs. spec's tool_wear) and missing modality-dropout regularizer, caught via a second documentation pass against the project README.
* **AI4I Task Definition Leak:** The AI4I dataset's `target_degradation` was trivially mapped to the input feature `tool_wear`, causing the model to learn a perfect identity function (Test MAE 4.56, but a Linear Regression baseline hit exactly 0.0000). Caught via a deliberate trivial-baseline stress test against the dataset card (which confirms AI4I has *no continuous RUL target*). The model training metrics looked perfect, which is exactly why this was dangerous. AI4I has been corrected to a classification-only task (binary failure + fault type), and the RUL regression head has been completely disabled for AI4I.
