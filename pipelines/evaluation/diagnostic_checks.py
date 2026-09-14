import torch
import torch.nn.functional as F
import numpy as np
import yaml
import json
import os
from ml.data.naman_adapter import NpzDataset, canonical_collate_fn
from torch.utils.data import DataLoader
from ml.models.mesh_model import MESHModel
from ml.models.ai4i_model import AI4IModel

device = "cpu"

print("=============================================")
print("CHECK 2 & 3: DataLoader Split Leakage & Masking")
print("=============================================")
# 2. Check Train.py's split implementation
print("In train.py, the DataLoader is currently implemented as:")
print("""
        # Since we only have a 20-sample fixture, we just split it into 16 train, 4 val for the mechanism check.
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
""")
print("This means the training run did NOT use the JSON splits. It randomly split the `train/data.npz` file 80/20.")

print("\nChecking raw mask density straight from the NPZ (AI4I train):")
ai4i_npz = np.load("data/processed/ai4i2020/train/data.npz")
total_masks = 0
total_zeros = 0
for k in ai4i_npz.files:
    if k.startswith("mask_"):
        total_masks += ai4i_npz[k].size
        total_zeros += (ai4i_npz[k] == 0).sum()
print(f"Raw NPZ Mask Zeros: {total_zeros}/{total_masks} ({(total_zeros/total_masks)*100:.1f}%)")

print("\n=============================================")
print("CHECK 1 & 4: Variance Collapse & Real-Unit MAE on TEST set")
print("=============================================")

print("\n--- CMAPSS TEST EVALUATION ---")
with open("configs/model/cmapss_model.yaml") as f:
    model_config = yaml.safe_load(f)

model_cmapss = MESHModel(
    native_modalities=model_config['native_modalities'],
    cnn_in_channels_map=model_config['modality_channels'],
    encoder_config=model_config['encoder_config'],
    fusion_config=model_config['fusion_config'],
    temporal_config=model_config['temporal_config'],
    heads_config=model_config['heads_config'],
    dropout_p=0.0
)
import sys
    ckpt_cmapss_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/cmapss/cmapss_v1_trained.pt"
    ckpt_cmapss = torch.load(ckpt_cmapss_path, map_location=device, weights_only=False)
if 'state_dict' in ckpt_cmapss:
    try:
        model_cmapss.load_state_dict(ckpt_cmapss['state_dict'])
        model_cmapss.eval()
    except Exception as e:
        print(e)
        
cmapss_mean = ckpt_cmapss['target_normalization']['rul_mean']
cmapss_std = ckpt_cmapss['target_normalization']['rul_std']

test_ds_cmapss = NpzDataset("data/processed/cmapss_fd001/test/data.npz", model_config['native_modalities'], "cmapss")
test_loader_cmapss = DataLoader(test_ds_cmapss, batch_size=32, collate_fn=lambda b: canonical_collate_fn(b, model_config['native_modalities']))

c_maes = []
c_variances = []
with torch.no_grad():
    for batch in test_loader_cmapss:
        preds, _ = model_cmapss(batch.modality_values, batch.modality_mask)
        pred_rul_mean = (preds['rul_mean'] * cmapss_std) + cmapss_mean
        y_true = batch.target_rul
        
        mae = torch.abs(pred_rul_mean.squeeze() - y_true).mean().item()
        c_maes.append(mae)
        
        if 'rul_variance' in preds:
            var_real = preds['rul_variance'] * (cmapss_std**2)
            c_variances.append(var_real.mean().item())

print(f"CMAPSS Test MAE: {np.mean(c_maes):.4f}")
if c_variances:
    print(f"CMAPSS Test Mean Variance: {np.mean(c_variances):.4f}")

from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix

print("\n--- AI4I TEST EVALUATION ---")
with open("configs/model/ai4i_model.yaml") as f:
    ai4i_model_config = yaml.safe_load(f)

model_ai4i = AI4IModel(
    native_modalities=ai4i_model_config['native_modalities'],
    modality_channels=ai4i_model_config['modality_channels'],
    embed_dim=ai4i_model_config['embed_dim'],
    num_heads=ai4i_model_config['num_heads'],
    num_fault_classes=ai4i_model_config['num_fault_classes'],
    enable_rul=ai4i_model_config.get('enable_rul', False),
    dropout_p=0.0
)
ckpt_ai4i_path = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/ai4i/ai4i_v1_trained.pt"
    ckpt_ai4i = torch.load(ckpt_ai4i_path, map_location=device, weights_only=False)
model_ai4i.load_state_dict(ckpt_ai4i['state_dict'])
model_ai4i.eval()

test_ds_ai4i = NpzDataset("data/processed/ai4i2020/test/data.npz", ai4i_model_config['native_modalities'], "ai4i")
test_loader = DataLoader(test_ds_ai4i, batch_size=32, collate_fn=lambda b: canonical_collate_fn(b, ai4i_model_config['native_modalities']))

total_samples = 0
correct_anomaly = 0
bce_losses = []

y_true_anomaly_all = []
y_pred_anomaly_all = []

y_true_fault_all = []
y_pred_fault_all = []

with torch.no_grad():
    for batch in test_loader:
        preds, _ = model_ai4i(batch.modality_values, batch.modality_mask)
        
        y_true_anom = batch.target_degradation # This maps to binary failure for AI4I
        y_true_fault = batch.target_fault_class
        
        if y_true_anom is not None and 'anomaly_logit' in preds:
            logits = preds['anomaly_logit']
            loss = F.binary_cross_entropy_with_logits(logits, y_true_anom).item()
            bce_losses.append(loss)
            
            probs = torch.sigmoid(logits)
            preds_bin = (probs > 0.5).float()
            
            y_true_anomaly_all.extend(y_true_anom.cpu().numpy().flatten())
            y_pred_anomaly_all.extend(preds_bin.cpu().numpy().flatten())
            
            total_samples += y_true_anom.size(0)
            
        if y_true_fault is not None and 'fault_logits' in preds:
            fault_logits = preds['fault_logits']
            fault_preds = torch.argmax(fault_logits, dim=1)
            y_true_fault_all.extend(y_true_fault.cpu().numpy().flatten())
            y_pred_fault_all.extend(fault_preds.cpu().numpy().flatten())

if total_samples > 0:
    y_true_anomaly_all = np.array(y_true_anomaly_all)
    y_pred_anomaly_all = np.array(y_pred_anomaly_all)
    
    num_failures = (y_true_anomaly_all == 1).sum()
    num_normal = (y_true_anomaly_all == 0).sum()
    print(f"Class Distribution: {num_failures} failures ({(num_failures/len(y_true_anomaly_all))*100:.1f}%), {num_normal} normal ({(num_normal/len(y_true_anomaly_all))*100:.1f}%)")
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_anomaly_all, y_pred_anomaly_all, labels=[1], zero_division=0)
    acc = (y_true_anomaly_all == y_pred_anomaly_all).mean() * 100
    
    print(f"AI4I Test BCE Loss: {np.mean(bce_losses):.4f}")
    print(f"AI4I Test Anomaly Accuracy: {acc:.2f}%")
    print(f"AI4I Test Anomaly Positive Class (Failure) - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}")
    
    if len(y_true_fault_all) > 0:
        y_true_fault_all = np.array(y_true_fault_all)
        y_pred_fault_all = np.array(y_pred_fault_all)
        fault_macro_f1 = f1_score(y_true_fault_all, y_pred_fault_all, average='macro', zero_division=0)
        print(f"AI4I Test Fault Classification Macro-F1: {fault_macro_f1:.4f}")
else:
    print("AI4I Test target_degradation not found.")
    
