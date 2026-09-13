import torch
import torch.nn.functional as F
import yaml
import numpy as np
from typing import Dict, Any

try:
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ml.models.mesh_model import MESHModel
from ml.data.mock_canonical_batch import generate_mock_canonical_batch

def compute_regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, y_var: torch.Tensor) -> Dict[str, Any]:
    mae = torch.abs(y_true - y_pred).mean().item()
    rmse = torch.sqrt(((y_true - y_pred) ** 2).mean()).item()
    
    # Calibration Check (Coverage)
    # y_var is the predicted variance. Std = sqrt(var)
    std = torch.sqrt(y_var)
    within_1_std = ((y_true >= y_pred - std) & (y_true <= y_pred + std)).float().mean().item()
    within_2_std = ((y_true >= y_pred - 2*std) & (y_true <= y_pred + 2*std)).float().mean().item()
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Coverage_1_std": within_1_std,
        "Coverage_2_std": within_2_std
    }

def compute_classification_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, is_binary: bool = False) -> Dict[str, Any]:
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not available"}
        
    metrics = {}
    
    if is_binary:
        y_pred_classes = (y_pred_probs > 0.5).astype(int)
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='binary', zero_division=0)
            metrics.update({"Precision": precision, "Recall": recall, "F1": f1})
            
            if len(np.unique(y_true)) > 1:
                metrics["ROC_AUC"] = roc_auc_score(y_true, y_pred_probs)
            else:
                metrics["ROC_AUC"] = "N/A - Only one class present in true labels"
        except Exception as e:
            metrics["ROC_AUC"] = f"N/A - Error: {str(e)}"
    else:
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='weighted', zero_division=0)
            metrics.update({"Precision": precision, "Recall": recall, "F1": f1})
            
            if len(np.unique(y_true)) > 1:
                # Multi-class AUC requires One-vs-Rest and probabilities for all classes
                metrics["ROC_AUC"] = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
            else:
                metrics["ROC_AUC"] = "N/A - Only one class present in true labels"
        except Exception as e:
             metrics["ROC_AUC"] = f"N/A - Error: {str(e)}"
             
    metrics["Accuracy"] = accuracy_score(y_true, y_pred_classes)
    return metrics

def evaluate_model(checkpoint_path: str):
    print(f"=== LOADING CHECKPOINT: {checkpoint_path} ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_config = checkpoint['model_config']
    train_config = checkpoint['train_config']
    norm_stats = checkpoint['target_normalization']
    rul_mean_stat = norm_stats['rul_mean']
    rul_std_stat = norm_stats['rul_std']
    
    print(f"Extracted Normalization Stats -> Mean: {rul_mean_stat}, Std: {rul_std_stat}")
    
    native_modalities = ["temperature", "tool_wear", "rotational_speed", "torque"]
    cnn_in_channels_map = {m: 1 for m in native_modalities}
    
    model = MESHModel(
        native_modalities=native_modalities,
        cnn_in_channels_map=cnn_in_channels_map,
        encoder_config=model_config['encoder'],
        fusion_config=model_config['fusion'],
        temporal_config=model_config['temporal'],
        heads_config=model_config['heads'],
        dropout_p=0.0 # Evaluation mode
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    # TODO: When Naman's real data lands, remove this mock generation. 
    # The evaluation must load a genuinely persisted test split (e.g., split by machine_id or time)
    # to avoid data leakage, rather than randomly generating a batch.
    print("\nGenerating fresh held-out test set (N=100) to guarantee non-overlap with training splits...")
    test_batch = generate_mock_canonical_batch(batch_size=100, window_size=20)
    
    modality_values = {k: v.to(device) for k, v in test_batch.modality_values.items()}
    modality_mask = test_batch.modality_mask.to(device)
    y_rul_true_real = test_batch.target_rul.to(device) # Real units
    y_fault_true = test_batch.target_fault_class.cpu().numpy()
    
    # Binarize anomaly since the mock target is uniform random [0,1],
    # and AUC/F1 require binary targets.
    y_anomaly_true = torch.round(test_batch.target_degradation).cpu().numpy()

    print("\n=== RUNNING EVALUATION ===")
    with torch.no_grad():
        preds, _ = model(modality_values, modality_mask)
        
        # 1. De-normalize RUL predictions to REAL units
        pred_rul_mean_real = (preds['rul_mean'] * rul_std_stat) + rul_mean_stat
        # Variance scales by std^2
        pred_rul_var_real = preds['rul_variance'] * (rul_std_stat ** 2)
        
        rul_metrics = compute_regression_metrics(y_rul_true_real, pred_rul_mean_real, pred_rul_var_real)
        
        # 2. Fault Metrics (Multi-class)
        pred_fault_probs = F.softmax(preds['fault_logits'], dim=1).cpu().numpy()
        fault_metrics = compute_classification_metrics(y_fault_true, pred_fault_probs, is_binary=False)
        
        # 3. Anomaly Metrics (Binary)
        pred_anomaly_probs = torch.sigmoid(preds['anomaly_logit']).cpu().numpy()
        anomaly_metrics = compute_classification_metrics(y_anomaly_true, pred_anomaly_probs, is_binary=True)
        
    print("\n[Regression] RUL Metrics (REAL UNITS - Cycles):")
    for k, v in rul_metrics.items():
        if isinstance(v, float):
             print(f"  {k}: {v:.4f}")
        else:
             print(f"  {k}: {v}")
             
    # --- DIAGNOSTICS ---
    print("\n--- DIAGNOSTICS ---")
    std = torch.sqrt(pred_rul_var_real)
    residuals = (y_rul_true_real - pred_rul_mean_real) / std
    print(f"Standardized Residuals - Min: {residuals.min().item():.4f}, Max: {residuals.max().item():.4f}, Mean: {residuals.mean().item():.4f}, Std: {residuals.std().item():.4f}")
    
    anom_probs = pred_anomaly_probs
    print(f"Anomaly Probs - Min: {anom_probs.min():.4f}, Max: {anom_probs.max():.4f}, Mean: {anom_probs.mean():.4f}")
    pred_pos = (anom_probs > 0.5).sum()
    print(f"Anomaly Predicted Positives: {pred_pos} / {len(anom_probs)}")
    
    fault_classes = np.argmax(pred_fault_probs, axis=1)
    unique, counts = np.unique(fault_classes, return_counts=True)
    fault_dist = dict(zip(unique, counts))
    print(f"Fault Predicted Class Distribution: {fault_dist}")
    print("-------------------")
             
    print("\n[Classification] Fault Metrics:")
    for k, v in fault_metrics.items():
        if isinstance(v, float):
             print(f"  {k}: {v:.4f}")
        else:
             print(f"  {k}: {v}")
             
    print("\n[Classification] Anomaly Detection Metrics:")
    for k, v in anomaly_metrics.items():
        if isinstance(v, float):
             print(f"  {k}: {v:.4f}")
        else:
             print(f"  {k}: {v}")

if __name__ == "__main__":
    evaluate_model("checkpoints/model_best.pt")
