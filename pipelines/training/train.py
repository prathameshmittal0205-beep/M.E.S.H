import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import datetime
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from ml.models.mesh_model import MESHModel
from pipelines.training.tracker import ExperimentTracker

PREV_STATS = {}
from ml.data.mock_canonical_batch import generate_mock_canonical_batch
from ml.data.contract import CanonicalBatch

# Mock Dataset until data pipeline is ready
class MockDataset(Dataset):
    def __init__(self, size=100, native_modalities=None):
        self.size = size
        self.native_modalities = native_modalities

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # We can just generate a batch of size 1 and strip the batch dimension
        batch = generate_mock_canonical_batch(batch_size=1, window_size=20, native_modalities=self.native_modalities)
        return {
            'modality_values': {k: v.squeeze(0) for k, v in batch.modality_values.items()},
            'modality_mask': batch.modality_mask.squeeze(0),
            'target_rul': batch.target_rul.squeeze(0),
            'target_fault_class': batch.target_fault_class.squeeze(0),
            'target_degradation': batch.target_degradation.squeeze(0)
        }

def collate_fn(batch_list):
    modality_values = {k: [] for k in batch_list[0]['modality_values'].keys()}
    modality_mask = []
    target_rul = []
    target_fault = []
    target_deg = []
    
    for b in batch_list:
        for k, v in b['modality_values'].items():
            modality_values[k].append(v)
        modality_mask.append(b['modality_mask'])
        target_rul.append(b['target_rul'])
        target_fault.append(b['target_fault_class'])
        target_deg.append(b['target_degradation'])
        
    return CanonicalBatch(
        sample_id=["mock" for _ in batch_list],
        source_dataset=["mock" for _ in batch_list],
        dataset_version=["v1" for _ in batch_list],
        run_id=["mock_run" for _ in batch_list],
        window_start=[0.0 for _ in batch_list],
        window_end=[20.0 for _ in batch_list],
        sampling_interval_seconds=[1.0 for _ in batch_list],
        native_modalities=batch_list[0].get('native_modalities', ["temperature", "tool_wear", "rotational_speed", "torque"]),
        preprocessor_version=["v1" for _ in batch_list],
        scaler_version=["v1" for _ in batch_list],
        modality_values={k: torch.stack(v) for k, v in modality_values.items()},
        modality_mask=torch.stack(modality_mask),
        target_rul=torch.stack(target_rul),
        target_fault_class=torch.stack(target_fault),
        target_degradation=torch.stack(target_deg)
    )

def main():
    print("=== INITIALIZING TRAINING PIPELINE ===")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", default="configs/training/cmapss_training.yaml")
    parser.add_argument("--model_config", default="configs/model/cmapss_model.yaml")
    args = parser.parse_args()
    
    with open(args.train_config, "r") as f:
        train_config = yaml.safe_load(f)
        
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    # Set seeds
    seed = train_config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    os.makedirs(train_config.get('checkpoint_dir', 'checkpoints/'), exist_ok=True)
    os.makedirs(train_config.get('log_dir', 'logs/'), exist_ok=True)
    
    native_modalities = model_config.get('native_modalities', [])
    if not native_modalities:
        native_modalities = ["temperature", "tool_wear", "rotational_speed", "torque"]
    model_type = model_config.get('model_type', 'cmapss')
    
    if model_type == 'AI4IModel' or model_type == 'ai4i':
        from ml.models.ai4i_model import AI4IModel
        model = AI4IModel(
            native_modalities=native_modalities,
            modality_channels=model_config['modality_channels'],
            embed_dim=model_config['embed_dim'],
            num_heads=model_config['num_heads'],
            num_fault_classes=model_config['num_fault_classes'],
            dropout_p=model_config['dropout_p']
        )
    else:
        # Fallback to older config keys if present
        modality_channels = model_config.get('modality_channels', {m: 1 for m in native_modalities})
        encoder_config = model_config.get('encoder_config', model_config.get('encoder'))
        fusion_config = model_config.get('fusion_config', model_config.get('fusion'))
        temporal_config = model_config.get('temporal_config', model_config.get('temporal'))
        heads_config = model_config.get('heads_config', model_config.get('heads'))
        
        model = MESHModel(
            native_modalities=native_modalities,
            cnn_in_channels_map=modality_channels,
            encoder_config=encoder_config,
            fusion_config=fusion_config,
            temporal_config=temporal_config,
            heads_config=heads_config,
            dropout_p=model_config['dropout_p']
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(train_config['learning_rate']),
        weight_decay=float(train_config['weight_decay'])
    )
    
    # Loss functions
    fault_criterion = nn.CrossEntropyLoss()
    anomaly_criterion = nn.BCEWithLogitsLoss()
    
    tracker = ExperimentTracker(train_config['log_dir'], train_config['experiment_name'])
    
    # Initialize Dataset
    if 'dataset_path' in train_config and 'dataset_name' in train_config:
        from ml.data.naman_adapter import NpzDataset, canonical_collate_fn
        from torch.utils.data import DataLoader
        
        train_path = train_config['dataset_path']
        val_path = train_path.replace('/train/', '/val/')
        
        print(f"Loading train dataset from {train_path}...")
        train_dataset = NpzDataset(train_path, model_config['native_modalities'], train_config['dataset_name'])
        
        print(f"Loading val dataset from {val_path}...")
        val_dataset = NpzDataset(val_path, model_config['native_modalities'], train_config['dataset_name'])
        
        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_config['batch_size'], 
            shuffle=True, 
            collate_fn=lambda b: canonical_collate_fn(b, model_config['native_modalities'])
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=train_config['batch_size'], 
            shuffle=False, 
            collate_fn=lambda b: canonical_collate_fn(b, model_config['native_modalities'])
        )
    else:
        print("WARNING: No dataset_path found in config. Falling back to synthetic mock data.")
        from ml.data.mock_canonical_batch import generate_mock_canonical_batch
        train_loader = [generate_mock_canonical_batch(batch_size=train_config['batch_size'], window_size=20, native_modalities=native_modalities) for _ in range(10)]
        val_loader = [generate_mock_canonical_batch(batch_size=train_config['batch_size'], window_size=20, native_modalities=native_modalities) for _ in range(2)]
    
    epochs = train_config['epochs']
    rul_mean_stat = train_config['target_normalization']['rul_mean']
    rul_std_stat = train_config['target_normalization']['rul_std']
    clip_grad = train_config.get('clip_grad', 1.0)
    
    print(f"Training on {device} for {epochs} epochs...")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        epoch_train_loss = 0.0
        epoch_rul_loss = 0.0
        epoch_fault_loss = 0.0
        epoch_anomaly_loss = 0.0
        epoch_rul_variance_sum = 0.0
        epoch_rul_mae_sum = 0.0
        epoch_rul_count = 0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move to device
            modality_values = {k: v.to(device) for k, v in batch.modality_values.items()}
            modality_mask = batch.modality_mask.to(device)
            y_rul = batch.target_rul.to(device) if batch.target_rul is not None else None
            y_fault = batch.target_fault_class.to(device) if batch.target_fault_class is not None else None
            y_anomaly = batch.target_degradation.to(device) if batch.target_degradation is not None else None
            
            # Modality Dropout (regularization during training)
            modality_dropout_p = train_config.get('modality_dropout_p', 0.0)
            if modality_dropout_p > 0.0:
                drop_mask = (torch.rand_like(modality_mask) > modality_dropout_p).float()
                
                # Apply dropout to the mask (values remain untouched)
                modality_mask = modality_mask * drop_mask
                
                # Prevent ALL modalities from being dropped for any sample in the final mask.
                # Even though nan_to_num handles forward pass NaNs, MHA backward pass computes 0 * NaN = NaN,
                # causing gradient explosion if a sample is fully masked.
                all_dropped = (modality_mask.sum(dim=1) == 0)
                if all_dropped.any():
                    # For each sample where all were dropped, restore one originally available modality
                    original_mask = batch.modality_mask.to(device)
                    for i in range(modality_mask.shape[0]):
                        if all_dropped[i]:
                            valid_indices = torch.nonzero(original_mask[i]).squeeze(-1)
                            if len(valid_indices) > 0:
                                rand_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))]
                                modality_mask[i, rand_idx] = 1.0
                            else:
                                modality_mask[i, 0] = 1.0
                
            # Forward pass
            preds, _ = model(modality_values, modality_mask)
            
            # Loss computation
            loss = 0.0
            w = train_config['loss_weights']
            
            # RUL Loss
            loss_rul = torch.tensor(0.0, device=device)
            if w.get('rul', 0) > 0 and 'rul_mean' in preds and batch.target_rul is not None:
                y_rul = batch.target_rul.to(device)
                y_rul_scaled = (y_rul - rul_mean_stat) / (rul_std_stat + 1e-6)
                if epoch < train_config.get('variance_warmup_epochs', 0):
                    loss_rul = F.mse_loss(preds['rul_mean'], y_rul_scaled)
                else:
                    loss_rul = (0.5 * (torch.log(preds['rul_variance']) + ((y_rul_scaled - preds['rul_mean'])**2 / preds['rul_variance']))).mean()
                loss += w['rul'] * loss_rul
                
            # Fault Loss
            loss_fault = torch.tensor(0.0, device=device)
            if w.get('fault', 0) > 0 and 'fault_logits' in preds and batch.target_fault_class is not None:
                y_fault = batch.target_fault_class.to(device)
                loss_fault = F.cross_entropy(preds['fault_logits'], y_fault)
                loss += w['fault'] * loss_fault
                
            # Anomaly Loss
            loss_anomaly = torch.tensor(0.0, device=device)
            if w.get('anomaly', 0) > 0 and 'anomaly_logit' in preds and batch.target_degradation is not None:
                y_anomaly = batch.target_degradation.to(device)
                loss_anomaly = F.binary_cross_entropy_with_logits(preds['anomaly_logit'], y_anomaly)
                loss += w['anomaly'] * loss_anomaly
            
            # DIAGNOSTIC CHECK
            if torch.isnan(loss):
                print(f"!!! NaN DETECTED at Epoch {epoch}, Step {step} !!!")
                print(f"loss_rul: {loss_rul.item()}, loss_fault: {loss_fault.item()}, loss_anomaly: {loss_anomaly.item()}")
                if 'rul_variance' in preds:
                    print(f"rul_variance min: {preds['rul_variance'].min().item()}, max: {preds['rul_variance'].max().item()}")
                if 'rul_mean' in preds:
                    print(f"rul_mean min: {preds['rul_mean'].min().item()}, max: {preds['rul_mean'].max().item()}")
                try:
                    print(f"PREV_STATS: {PREV_STATS}")
                except NameError:
                    pass
                
                # Let's also check if any model parameters are NaN
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"Parameter {name} has NaNs!")
                raise ValueError("NaN loss encountered")
            
            # We also want to capture the step immediately before NaN, but since we don't know when it happens until it does,
            # we can store the previous step's stats.
            PREV_STATS = {
                'epoch': epoch,
                'step': step,
                'loss_rul': loss_rul.item(),
                'loss_fault': loss_fault.item(),
                'loss_anomaly': loss_anomaly.item()
            }
            if 'rul_variance' in preds:
                PREV_STATS['rul_variance_min'] = preds['rul_variance'].min().item()
            if 'rul_mean' in preds:
                PREV_STATS['rul_mean_max'] = preds['rul_mean'].max().item()
            
            loss.backward()
            
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
            optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_rul_loss += loss_rul.item()
            epoch_fault_loss += loss_fault.item()
            epoch_anomaly_loss += loss_anomaly.item()
            
            tracker.log_metrics({
                'loss/total': loss.item(),
                'loss/rul': loss_rul.item(),
                'loss/fault': loss_fault.item(),
                'loss/anomaly': loss_anomaly.item()
            }, global_step, prefix="train")
            global_step += 1
            
            if 'rul_mean' in preds and batch.target_rul is not None:
                # Accumulate for epoch average
                epoch_rul_variance_sum += preds['rul_variance'].mean().item()
                epoch_rul_mae_sum += torch.abs((preds['rul_mean'] * rul_std_stat + rul_mean_stat).squeeze() - batch.target_rul.squeeze().to(device)).mean().item()
                epoch_rul_count += 1
                
        # Calculate epoch averages
        avg_train_loss = epoch_train_loss / len(train_loader)
        if epoch_rul_count > 0:
            avg_train_var = epoch_rul_variance_sum / epoch_rul_count
            avg_train_mae = epoch_rul_mae_sum / epoch_rul_count
        else:
            avg_train_var = 0.0
            avg_train_mae = 0.0
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                modality_values = {k: v.to(device) for k, v in batch.modality_values.items()}
                modality_mask = batch.modality_mask.to(device)
                y_rul = batch.target_rul.to(device) if batch.target_rul is not None else None
                y_fault = batch.target_fault_class.to(device) if batch.target_fault_class is not None else None
                y_anomaly = batch.target_degradation.to(device) if batch.target_degradation is not None else None
                
                preds, _ = model(modality_values, modality_mask)
                
                # Validation Loss computation
                loss = 0.0
                
                loss_rul = torch.tensor(0.0, device=device)
                if w.get('rul', 0) > 0 and 'rul_mean' in preds and batch.target_rul is not None:
                    y_rul_scaled = (y_rul - rul_mean_stat) / (rul_std_stat + 1e-6)
                    if epoch < train_config.get('variance_warmup_epochs', 0):
                        loss_rul = F.mse_loss(preds['rul_mean'], y_rul_scaled)
                    else:
                        loss_rul = (0.5 * (torch.log(preds['rul_variance']) + ((y_rul_scaled - preds['rul_mean'])**2 / preds['rul_variance']))).mean()
                    loss += w['rul'] * loss_rul
                    
                loss_fault = torch.tensor(0.0, device=device)
                if w.get('fault', 0) > 0 and 'fault_logits' in preds and batch.target_fault_class is not None:
                    loss_fault = F.cross_entropy(preds['fault_logits'], y_fault)
                    loss += w['fault'] * loss_fault
                    
                loss_anomaly = torch.tensor(0.0, device=device)
                if w.get('anomaly', 0) > 0 and 'anomaly_logit' in preds and batch.target_degradation is not None:
                    loss_anomaly = F.binary_cross_entropy_with_logits(preds['anomaly_logit'], y_anomaly)
                    loss += w['anomaly'] * loss_anomaly
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        avg_rul_loss = epoch_rul_loss / len(train_loader)
        avg_fault_loss = epoch_fault_loss / len(train_loader)
        avg_anomaly_loss = epoch_anomaly_loss / len(train_loader)
        
        # Diagnostic prints for Variance and MAE
        components_str = f"[RUL: {avg_rul_loss:.4f} | Fault: {avg_fault_loss:.4f} | Anomaly: {avg_anomaly_loss:.4f}]"
        if epoch_rul_count > 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} {components_str} | Val Loss: {avg_val_loss:.4f} | Train MAE: {avg_train_mae:.4f} | Train Var: {avg_train_var:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} {components_str} | Val Loss: {avg_val_loss:.4f}")
        
        tracker.log_metrics({'loss/total': avg_val_loss}, epoch, prefix="val")
        
        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Generate a version string based on training time
            model_version = f"v1.0-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model_config,
                'train_config': train_config,
                'model_version': model_version,
                'target_normalization': train_config.get('target_normalization', {'rul_mean': 0.0, 'rul_std': 1.0}),
                'metrics': {
                    'best_val_loss': best_val_loss,
                    'epoch': epoch
                }
            }
            checkpoint_dir = train_config.get('checkpoint_dir', 'checkpoints/')
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(checkpoint_dir, "model_best.pt"))
            
    print("Training Complete. Best model saved.")
    tracker.close()

if __name__ == "__main__":
    main()
