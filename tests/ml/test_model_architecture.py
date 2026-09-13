import pytest
import torch
import copy
from ml.models.mesh_model import MESHModel

@pytest.fixture
def base_config():
    return {
        "native_modalities": ["temperature", "tool_wear", "rotational_speed", "torque"],
        "cnn_in_channels_map": {"temperature": 1, "tool_wear": 1, "rotational_speed": 1, "torque": 1},
        "encoder_config": {
            "cnn_out_channels": 8,
            "cnn_kernel_size": 3,
            "bilstm_hidden_size": 16,
            "bilstm_num_layers": 1
        },
        "fusion_config": {"embed_dim": 32, "num_heads": 2, "dropout": 0.1},
        "temporal_config": {"d_model": 32, "nhead": 2, "num_layers": 1, "dim_feedforward": 16, "dropout": 0.1},
        "heads_config": {"rul_hidden_dim": 16, "fault_hidden_dim": 16, "anomaly_hidden_dim": 16, "num_fault_classes": 3},
        "dropout_p": 0.1
    }

@pytest.fixture
def model(base_config):
    return MESHModel(**base_config)

@pytest.fixture
def sample_batch():
    B = 2
    seq_len = 20
    mod_values = {
        "temperature": torch.randn(B, seq_len, 1),
        "tool_wear": torch.randn(B, seq_len, 1),
        "rotational_speed": torch.randn(B, seq_len, 1),
        "torque": torch.randn(B, seq_len, 1),
    }
    mask = torch.ones(B, 4)
    return mod_values, mask

def test_head_shapes(model, sample_batch):
    model.eval()
    mod_values, mask = sample_batch
    preds, _ = model(mod_values, mask)
    
    assert "rul_mean" in preds
    assert "rul_variance" in preds
    assert "anomaly_logit" in preds
    assert "fault_logits" in preds
    
    B = mod_values["temperature"].shape[0]
    
    assert preds["rul_mean"].shape == (B, 1)
    assert preds["rul_variance"].shape == (B, 1)
    assert preds["anomaly_logit"].shape == (B, 1)
    assert preds["fault_logits"].shape == (B, 3) # 3 fault classes

def test_masked_modality_zeroing(model, sample_batch):
    model.eval()
    mod_values, mask = sample_batch
    
    # Mask out tool_wear (index 1) for sample 0
    mask[0, 1] = 0.0
    
    with torch.no_grad():
        preds, attn = model(mod_values, mask)
        
    assert attn is not None
    # Wait, the attention mask logic zeros out the output of the fusion layer.
    # To test exactly that the modality's query was zeroed, we inspect the fusion layer's internal logic,
    # or we can test that attn is uniform for the masked row (before zeroing), 
    # but the fusion layer zeros out the output `masked_out = attn_out * expanded_mask`.
    # We can check model.fusion._last_masked_out since we stored it during development!
    
    assert hasattr(model.fusion, "_last_masked_out"), "Model must expose _last_masked_out for test verification."
    masked_out = model.fusion._last_masked_out
    # masked_out shape: [B * seq_len, num_modalities, embed_dim]
    # B=2, seq_len=20 -> [40, 4, 32]. Sample 0 is indices 0-19. Modality 1 (tool_wear).
    sample_0_vib = masked_out[0:20, 1, :]
    assert torch.all(sample_0_vib == 0.0)

def test_all_modalities_masked_no_nan(model, sample_batch):
    model.eval()
    mod_values, mask = sample_batch
    
    # Mask ALL modalities for sample 0
    mask[0, :] = 0.0
    
    with torch.no_grad():
        preds, _ = model(mod_values, mask)
        
    assert not torch.isnan(preds["rul_mean"]).any()
    assert not torch.isnan(preds["rul_variance"]).any()
    assert not torch.isnan(preds["anomaly_logit"]).any()
    assert not torch.isnan(preds["fault_logits"]).any()

def test_eval_determinism(model, sample_batch):
    model.eval()
    mod_values, mask = sample_batch
    
    with torch.no_grad():
        preds1, _ = model(mod_values, mask)
        preds2, _ = model(mod_values, mask)
        
    assert torch.allclose(preds1["rul_mean"], preds2["rul_mean"])
    assert torch.allclose(preds1["rul_variance"], preds2["rul_variance"])

def test_train_non_determinism(model, sample_batch):
    # Confirms dropout is actually active (no seed reset)
    model.train()
    mod_values, mask = sample_batch
    
    with torch.no_grad():
        preds1, _ = model(mod_values, mask)
        preds2, _ = model(mod_values, mask)
        
    # With dropout_p=0.1, multiple passes in train mode should produce slightly different results
    assert not torch.allclose(preds1["rul_mean"], preds2["rul_mean"])

def test_train_fixed_seed_determinism(model, sample_batch):
    # Confirms the model is completely deterministic when seeded (required for reproducibility)
    model.train()
    mod_values, mask = sample_batch
    
    torch.manual_seed(42)
    with torch.no_grad():
        preds1, _ = model(mod_values, mask)
        
    torch.manual_seed(42)
    with torch.no_grad():
        preds2, _ = model(mod_values, mask)
        
    assert torch.allclose(preds1["rul_mean"], preds2["rul_mean"])
    assert torch.allclose(preds1["rul_variance"], preds2["rul_variance"])
