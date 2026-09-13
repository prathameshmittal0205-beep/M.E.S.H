import os
import json
import torch
import copy
from datetime import datetime

from ml.models.mesh_model import MESHModel
from ml.models.baselines import ConcatenationBaseline
from ml.data.mock_canonical_batch import generate_mock_canonical_batch

def compute_rul_metrics(y_true, y_pred, y_var, std_stat, mean_stat):
    # De-normalize
    y_pred_real = (y_pred * std_stat) + mean_stat
    y_var_real = y_var * (std_stat ** 2)
    
    mae = torch.abs(y_true - y_pred_real).mean().item()
    std = torch.sqrt(y_var_real)
    coverage_1_std = ((y_true >= y_pred_real - std) & (y_true <= y_pred_real + std)).float().mean().item()
    mean_var = y_var_real.mean().item()
    
    return {
        "MAE_real_units": round(mae, 4),
        "Coverage_1_std": round(coverage_1_std, 4),
        "Mean_Variance_real_units": round(mean_var, 4)
    }

def main():
    print("=== MESH Missing-Modality Robustness Experiment ===")
    print("CAVEAT: The models and baseline are currently evaluated on MOCK uniform noise.")
    print("Any 'degradation' trend observed here is an artifact of the mock data and untrained states.")
    print("This run exclusively verifies PIPELINE CORRECTNESS. Real missing-modality robustness")
    print("conclusions await a properly trained model on Naman's real sensor dataset.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load trained MESHModel
    checkpoint_path = "ml/checkpoints/model_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']
    target_norm = checkpoint['target_normalization']
    rul_mean_stat, rul_std_stat = target_norm['rul_mean'], target_norm['rul_std']
    
    native_modalities = ["temperature", "tool_wear", "rotational_speed", "torque"]
    cnn_in_channels_map = {m: 1 for m in native_modalities}
    
    mesh_model = MESHModel(
        native_modalities=native_modalities,
        cnn_in_channels_map=cnn_in_channels_map,
        encoder_config=model_config['encoder'],
        fusion_config=model_config['fusion'],
        temporal_config=model_config['temporal'],
        heads_config=model_config['heads'],
        dropout_p=0.0
    )
    mesh_model.load_state_dict(checkpoint['state_dict'])
    mesh_model.to(device)
    mesh_model.eval()

    # 2. Load ConcatenationBaseline (UNTRAINED)
    # The baseline has not been trained on mock data yet. We initialize it to verify
    # the architectural degradation comparison pipeline works.
    baseline = ConcatenationBaseline(
        native_modalities=native_modalities,
        cnn_in_channels_map=cnn_in_channels_map,
        encoder_config=model_config['encoder'],
        temporal_config=model_config['temporal'],
        heads_config=model_config['heads'],
        use_mask=True
    )
    baseline.to(device)
    baseline.eval()
    
    # 3. Generate a SINGLE FIXED held-out test batch
    # Re-used across all masking conditions to eliminate sample variation.
    test_batch = generate_mock_canonical_batch(batch_size=100, window_size=20, native_modalities=native_modalities)
    base_values = {k: v.to(device) for k, v in test_batch.modality_values.items()}
    y_rul_true = test_batch.target_rul.to(device)
    
    # Define masking conditions (indices corresponding to native_modalities)
    # 0: temp, 1: vib, 2: speed, 3: torque
    conditions = {
        "0_dropped_baseline": [],
        "1_dropped_temperature": [0],
        "1_dropped_tool_wear": [1],
        "1_dropped_rotational_speed": [2],
        "1_dropped_torque": [3],
        "2_dropped_temp_wear": [0, 1],
        "2_dropped_speed_torque": [2, 3],
        "4_dropped_total_dropout": [0, 1, 2, 3]
    }
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "caveat": "MOCK DATA EVALUATION ONLY. BASELINE IS UNTRAINED.",
            "test_set_size": 100,
            "modalities": native_modalities
        },
        "experiments": []
    }
    
    with torch.no_grad():
        for cond_name, dropped_indices in conditions.items():
            print(f"Testing Condition: {cond_name}")
            # Create a clean mask (all 1s) and zero out dropped modalities
            cond_mask = torch.ones(100, len(native_modalities)).to(device)
            for idx in dropped_indices:
                cond_mask[:, idx] = 0.0
                
            # MESH Prediction
            mesh_preds, _ = mesh_model(base_values, cond_mask)
            mesh_metrics = compute_rul_metrics(
                y_rul_true, mesh_preds['rul_mean'], mesh_preds['rul_variance'],
                rul_std_stat, rul_mean_stat
            )
            
            # Baseline Prediction
            base_preds, _ = baseline(base_values, cond_mask)
            base_metrics = compute_rul_metrics(
                y_rul_true, base_preds['rul_mean'], base_preds['rul_variance'],
                rul_std_stat, rul_mean_stat
            )
            
            results["experiments"].append({
                "condition": cond_name,
                "dropped_indices": dropped_indices,
                "percent_missing": f"{(len(dropped_indices)/4)*100}%",
                "mesh_model_metrics": mesh_metrics,
                "untrained_baseline_metrics": base_metrics
            })
            
    # Dump to JSON
    os.makedirs("pipelines/evaluation/results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"pipelines/evaluation/results/missing_modality_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nExperiment complete. Results saved to: {out_file}")
    print("Snapshot of Total Dropout condition vs Baseline:")
    print(json.dumps(results["experiments"][-1], indent=2))

if __name__ == "__main__":
    main()
