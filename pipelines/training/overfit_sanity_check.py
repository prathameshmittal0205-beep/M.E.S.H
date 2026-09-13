import torch
import torch.nn.functional as F
import yaml
from ml.models.mesh_model import MESHModel
from ml.data.mock_canonical_batch import generate_mock_canonical_batch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open("configs/model/da1_model.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    
    # Force single batch, no dropout
    model_config['fusion']['modality_dropout'] = 0.0
    
    native_modalities = ['temperature', 'tool_wear', 'rotational_speed', 'torque']
    model = MESHModel(
        native_modalities=native_modalities,
        encoder_config=model_config['encoder'],
        fusion_config=model_config['fusion'],
        temporal_config=model_config['temporal'],
        heads_config=model_config['heads'],
        cnn_in_channels_map={k: 1 for k in native_modalities}
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate exactly ONE batch
    batch = generate_mock_canonical_batch(batch_size=4, window_size=20, native_modalities=native_modalities)
    
    modality_values = {k: v.to(device) for k, v in batch.modality_values.items()}
    modality_mask = batch.modality_mask.to(device)
    y_rul = batch.target_rul.to(device)
    
    # Scale targets
    rul_mean_stat = 0.0
    rul_std_stat = 50.0
    y_rul_scaled = (y_rul - rul_mean_stat) / (rul_std_stat + 1e-6)
    
    print("=== OVERFIT SANITY CHECK ===")
    print(f"Target scaled values: {y_rul_scaled.tolist()}")
    
    model.train()
    for step in range(300):
        optimizer.zero_grad()
        preds, _ = model(modality_values, modality_mask)
        
        # NLL Loss
        loss = (0.5 * (torch.log(preds['rul_variance']) + ((y_rul_scaled - preds['rul_mean'])**2 / preds['rul_variance']))).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0 or step == 299:
            mae = torch.abs(preds['rul_mean'] - y_rul_scaled).mean().item()
            var = preds['rul_variance'].mean().item()
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | Mean MAE: {mae:.4f} | Mean Variance: {var:.4f}")

if __name__ == "__main__":
    main()
