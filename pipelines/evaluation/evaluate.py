import torch
import torch.nn.functional as F
import yaml
import numpy as np
from typing import Dict, Any
import argparse
from torch.utils.data import DataLoader
from ml.data.naman_adapter import NpzDataset, canonical_collate_fn

try:
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ml.models.mesh_model import MESHModel
from ml.models.ai4i_model import AI4IModel
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

def evaluate_model(checkpoint_path: str, dataset_path: str, dataset_name: str):
    print(f"=== LOADING CHECKPOINT: {checkpoint_path} ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_config = checkpoint['model_config']
    train_config = checkpoint['train_config']
    norm_stats = checkpoint['target_normalization']
    rul_mean_stat = norm_stats['rul_mean']
    rul_std_stat = norm_stats['rul_std']
    
    print(f"Extracted Normalization Stats -> Mean: {rul_mean_stat}, Std: {rul_std_stat}")
    
    native_modalities = model_config.get('native_modalities', ["temperature", "tool_wear", "rotational_speed", "torque"])
    cnn_in_channels_map = {m: 1 for m in native_modalities}
    
    if dataset_name == 'ai4i':
        model = AI4IModel(
            native_modalities=native_modalities,
            modality_channels=model_config['modality_channels'],
            embed_dim=model_config['embed_dim'],
            num_heads=model_config['num_heads'],
            num_fault_classes=model_config['num_fault_classes'],
            dropout_p=0.0
        )
    else:
        encoder_config = model_config.get('encoder_config', model_config.get('encoder'))
        fusion_config = model_config.get('fusion_config', model_config.get('fusion'))
        temporal_config = model_config.get('temporal_config', model_config.get('temporal'))
        heads_config = model_config.get('heads_config', model_config.get('heads'))
        modality_channels = model_config.get('modality_channels', cnn_in_channels_map)
        
        model = MESHModel(
            native_modalities=native_modalities,
            cnn_in_channels_map=modality_channels,
            encoder_config=encoder_config,
            fusion_config=fusion_config,
            temporal_config=temporal_config,
            heads_config=heads_config,
            dropout_p=0.0 # Evaluation mode
        )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    print(f"\nLoading test dataset from {dataset_path}...")
    test_dataset = NpzDataset(dataset_path, native_modalities, dataset_name)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=128, 
        shuffle=False, 
        collate_fn=lambda b: canonical_collate_fn(b, native_modalities)
    )

    y_rul_true_real_list = []
    y_fault_true_list = []
    y_anomaly_true_list = []
    
    pred_rul_mean_real_list = []
    pred_rul_var_real_list = []
    pred_fault_probs_list = []
    pred_anomaly_probs_list = []

    print("\n=== RUNNING EVALUATION ===")
    with torch.no_grad():
        for test_batch in test_loader:
            modality_values = {k: v.to(device) for k, v in test_batch.modality_values.items()}
            modality_mask = test_batch.modality_mask.to(device)
            
            preds, _ = model(modality_values, modality_mask)
            
            if test_batch.target_rul is not None:
                y_rul_true_real_list.append(test_batch.target_rul.cpu())
                pred_rul_mean_real_list.append(((preds['rul_mean'] * rul_std_stat) + rul_mean_stat).cpu())
                pred_rul_var_real_list.append((preds['rul_variance'] * (rul_std_stat ** 2)).cpu())
            
            if test_batch.target_fault_class is not None:
                y_fault_true_list.append(test_batch.target_fault_class.cpu().numpy())
                pred_fault_probs_list.append(F.softmax(preds['fault_logits'], dim=1).cpu().numpy())
                
            if test_batch.target_degradation is not None:
                y_anomaly_true_list.append(test_batch.target_degradation.cpu().numpy())
                pred_anomaly_probs_list.append(torch.sigmoid(preds['anomaly_logit']).cpu().numpy())
                
    if len(y_rul_true_real_list) > 0:
        y_rul_true_real = torch.cat(y_rul_true_real_list)
        pred_rul_mean_real = torch.cat(pred_rul_mean_real_list)
        pred_rul_var_real = torch.cat(pred_rul_var_real_list)
        rul_metrics = compute_regression_metrics(y_rul_true_real, pred_rul_mean_real, pred_rul_var_real)
    else:
        rul_metrics = {}
        y_rul_true_real = torch.zeros(0)
        pred_rul_mean_real = torch.zeros(0)
        pred_rul_var_real = torch.zeros(0)
        
    if len(y_fault_true_list) > 0:
        y_fault_true = np.concatenate(y_fault_true_list)
        pred_fault_probs = np.concatenate(pred_fault_probs_list)
        fault_metrics = compute_classification_metrics(y_fault_true, pred_fault_probs, is_binary=False)
    else:
        fault_metrics = {}
        pred_fault_probs = np.zeros((0, 1))
        
    if len(y_anomaly_true_list) > 0:
        y_anomaly_true = np.concatenate(y_anomaly_true_list)
        pred_anomaly_probs = np.concatenate(pred_anomaly_probs_list)
        anomaly_metrics = compute_classification_metrics(y_anomaly_true, pred_anomaly_probs, is_binary=True)
    else:
        anomaly_metrics = {}
        pred_anomaly_probs = np.zeros(0)

    print("\\n[Regression] RUL Metrics (REAL UNITS - Cycles):")
    if rul_metrics:
        for k, v in rul_metrics.items():
            if isinstance(v, float):
                 print(f"  {k}: {v:.4f}")
            else:
                 print(f"  {k}: {v}")
    else:
        print("  (No RUL targets)")

    print("\\n[Classification] Fault Metrics:")
    if fault_metrics:
        for k, v in fault_metrics.items():
            if isinstance(v, float):
                 print(f"  {k}: {v:.4f}")
            else:
                 print(f"  {k}: {v}")
    else:
        print("  (No Fault targets)")

    print("\\n[Classification] Anomaly Metrics:")
    if anomaly_metrics:
        for k, v in anomaly_metrics.items():
            if isinstance(v, float):
                 print(f"  {k}: {v:.4f}")
            else:
                 print(f"  {k}: {v}")
    else:
        print("  (No Anomaly targets)")

    print("\\n--- DIAGNOSTICS ---")
    if len(y_rul_true_real) > 0:
        std = torch.sqrt(pred_rul_var_real)
        residuals = (y_rul_true_real - pred_rul_mean_real) / std
        print(f"Standardized Residuals - Min: {residuals.min().item():.4f}, Max: {residuals.max().item():.4f}, Mean: {residuals.mean().item():.4f}, Std: {residuals.std().item():.4f}")
    
    if len(pred_anomaly_probs) > 0:
        anom_probs = pred_anomaly_probs
        print(f"Anomaly Probs - Min: {anom_probs.min():.4f}, Max: {anom_probs.max():.4f}, Mean: {anom_probs.mean():.4f}")
    
    if len(pred_fault_probs) > 0:
        fault_probs = pred_fault_probs.max(axis=1)
        print(f"Max Fault Probs (Confidence) - Min: {fault_probs.min():.4f}, Max: {fault_probs.max():.4f}, Mean: {fault_probs.mean():.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    args = parser.parse_args()
    evaluate_model(args.checkpoint, args.dataset_path, args.dataset_name)
