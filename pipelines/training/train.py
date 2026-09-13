import os
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
    with open("configs/training/da1_training.yaml", "r") as f:
        train_config = yaml.safe_load(f)
        
    with open("configs/model/da1_model.yaml", "r") as f:
        model_config = yaml.safe_load(f)

    # Set seeds
    seed = train_config['seed']
    torch.manual_seed(seed)
    
    native_modalities = ["temperature", "tool_wear", "rotational_speed", "torque"]
    cnn_in_channels_map = {m: 1 for m in native_modalities}
    
    model = MESHModel(
        native_modalities=native_modalities,
        cnn_in_channels_map=cnn_in_channels_map,
        encoder_config=model_config['encoder'],
        fusion_config=model_config['fusion'],
        temporal_config=model_config['temporal'],
        heads_config=model_config['heads'],
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
    
    dataset = MockDataset(size=200, native_modalities=native_modalities)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    epochs = train_config['epochs']
    rul_mean_stat = train_config['target_normalization']['rul_mean']
    rul_std_stat = train_config['target_normalization']['rul_std']
    clip_grad = train_config['clip_grad_norm']
    
    print(f"Training on {device} for {epochs} epochs...")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move to device
            modality_values = {k: v.to(device) for k, v in batch.modality_values.items()}
            modality_mask = batch.modality_mask.to(device)
            y_rul = batch.target_rul.to(device)
            y_fault = batch.target_fault_class.to(device)
            y_anomaly = batch.target_degradation.to(device)
            
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
            y_rul = batch.target_rul.to(device)
            y_fault = batch.target_fault_class.to(device)
            y_anomaly = batch.target_degradation.to(device)
            
            # RUL NLL Loss with scaled targets
            y_rul_scaled = (y_rul - rul_mean_stat) / (rul_std_stat + 1e-6)
            loss_rul = (0.5 * (torch.log(preds['rul_variance']) + ((y_rul_scaled - preds['rul_mean'])**2 / preds['rul_variance']))).mean()
            
            # Fault CrossEntropy Loss
            loss_fault = F.cross_entropy(preds['fault_logits'], y_fault)
            
            # Anomaly BCE Loss
            loss_anomaly = F.binary_cross_entropy_with_logits(preds['anomaly_logit'], y_anomaly)
            
            # Weighted Total Loss
            w = train_config['loss_weights']
            loss = w['rul'] * loss_rul + w['fault'] * loss_fault + w['anomaly'] * loss_anomaly
            
            # DIAGNOSTIC CHECK
            if torch.isnan(loss):
                print(f"!!! NaN DETECTED at Epoch {epoch}, Step {step} !!!")
                print(f"loss_rul: {loss_rul.item()}, loss_fault: {loss_fault.item()}, loss_anomaly: {loss_anomaly.item()}")
                print(f"rul_variance min: {preds['rul_variance'].min().item()}, max: {preds['rul_variance'].max().item()}")
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
                'rul_variance_min': preds['rul_variance'].min().item(),
                'rul_mean_min': preds['rul_mean'].min().item(),
                'rul_mean_max': preds['rul_mean'].max().item(),
            }
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            
            train_loss += loss.item()
            tracker.log_metrics({
                'loss/total': loss.item(),
                'loss/rul': loss_rul.item(),
                'loss/fault': loss_fault.item(),
                'loss/anomaly': loss_anomaly.item()
            }, global_step, prefix="train")
            global_step += 1
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                modality_values = {k: v.to(device) for k, v in batch.modality_values.items()}
                modality_mask = batch.modality_mask.to(device)
                y_rul = batch.target_rul.to(device)
                y_fault = batch.target_fault_class.to(device)
                y_anomaly = batch.target_degradation.to(device)
                
                preds, _ = model(modality_values, modality_mask)
                y_rul_scaled = (y_rul - rul_mean_stat) / (rul_std_stat + 1e-6)
                loss_rul = (0.5 * (torch.log(preds['rul_variance']) + ((y_rul_scaled - preds['rul_mean'])**2 / preds['rul_variance']))).mean()
                loss_fault = F.cross_entropy(preds['fault_logits'], y_fault)
                loss_anomaly = F.binary_cross_entropy_with_logits(preds['anomaly_logit'], y_anomaly)
                
                loss = w['rul'] * loss_rul + w['fault'] * loss_fault + w['anomaly'] * loss_anomaly
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        tracker.log_metrics({'loss/total': avg_val_loss}, epoch, prefix="val")
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
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
            os.makedirs(train_config['checkpoint_dir'], exist_ok=True)
            torch.save(checkpoint, os.path.join(train_config['checkpoint_dir'], "model_best.pt"))
            
    print("Training Complete. Best model saved.")
    tracker.close()

if __name__ == "__main__":
    main()
