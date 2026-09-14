import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import yaml
import sys

def run_baseline(config_path, dataset_name):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    train_path = config['dataset_path']
    val_path = train_path.replace('/train/', '/val/')
    test_path = train_path.replace('/train/', '/test/')
    
    print(f"\n=== Baseline for {dataset_name.upper()} ===")
    try:
        train_data = np.load(train_path)
        test_data = np.load(test_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"Train samples: {train_data['target_rul' if dataset_name == 'cmapss' else 'target_degradation'].shape}")
    print(f"Test samples: {test_data['target_rul' if dataset_name == 'cmapss' else 'target_degradation'].shape}")
    
    target_key = 'target_rul' if dataset_name == 'cmapss' else 'target_degradation'
    train_y = train_data[target_key]
    test_y = test_data[target_key]
    
    # Flatten modalities into a single feature vector
    train_X = []
    test_X = []
    
    modalities = [k for k in train_data.files if k.startswith('modality_')]
    for mod in modalities:
        # shape [N, T, C] or [N, C]. We flatten T*C for baseline
        tr_mod = train_data[mod].reshape(train_data[mod].shape[0], -1)
        te_mod = test_data[mod].reshape(test_data[mod].shape[0], -1)
        train_X.append(tr_mod)
        test_X.append(te_mod)
        
        # Test per-modality LR
        lr_mod = LinearRegression()
        lr_mod.fit(tr_mod, train_y)
        preds_mod = lr_mod.predict(te_mod)
        mae_mod = mean_absolute_error(test_y, preds_mod)
        print(f"Linear Regression on {mod} Test MAE: {mae_mod:.4f}")
        
    train_X = np.concatenate(train_X, axis=1)
    test_X = np.concatenate(test_X, axis=1)
    
    print(f"Train X shape: {train_X.shape}, Train Y shape: {train_y.shape}")
    print(f"Test X shape: {test_X.shape}, Test Y shape: {test_y.shape}")
    
    # Check simple mean distribution
    print(f"Test Y Min: {test_y.min():.2f}, Max: {test_y.max():.2f}, Mean: {test_y.mean():.2f}, Std: {test_y.std():.2f}")
    
    lr = LinearRegression()
    lr.fit(train_X, train_y)
    
    preds = lr.predict(test_X)
    mae = mean_absolute_error(test_y, preds)
    print(f"Linear Regression Test MAE: {mae:.4f}")

run_baseline('configs/training/cmapss_training.yaml', 'cmapss')
run_baseline('configs/training/ai4i_training.yaml', 'ai4i')
