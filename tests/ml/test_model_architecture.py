import pytest
import torch
import copy
from ml.models.mesh_model import MESHModel
from ml.models.ai4i_model import AI4IModel

# --- CMAPSS (Temporal) Fixtures ---

@pytest.fixture
def cmapss_config():
    return {
        "native_modalities": ["temperatures", "pressures", "speeds", "gas_flow", "operational_settings"],
        "cnn_in_channels_map": {"temperatures": 4, "pressures": 5, "speeds": 6, "gas_flow": 6, "operational_settings": 3},
        "encoder_config": {
            "cnn_out_channels": 8,
            "cnn_kernel_size": 3,
            "bilstm_hidden_size": 16,
            "bilstm_num_layers": 1
        },
        "fusion_config": {"embed_dim": 32, "num_heads": 2, "dropout": 0.1},
        "temporal_config": {"d_model": 32, "nhead": 2, "num_layers": 1, "dim_feedforward": 16, "dropout": 0.1},
        "heads_config": {"rul_hidden_dim": 16}, # No fault/anomaly
        "dropout_p": 0.1
    }

@pytest.fixture
def cmapss_model(cmapss_config):
    return MESHModel(**cmapss_config)

@pytest.fixture
def cmapss_sample_batch():
    B = 2
    seq_len = 20
    mod_values = {
        "temperatures": torch.randn(B, seq_len, 4),
        "pressures": torch.randn(B, seq_len, 5),
        "speeds": torch.randn(B, seq_len, 6),
        "gas_flow": torch.randn(B, seq_len, 6),
        "operational_settings": torch.randn(B, seq_len, 3),
    }
    mask = torch.ones(B, 5)
    return mod_values, mask

# --- AI4I (Static) Fixtures ---

@pytest.fixture
def ai4i_config():
    return {
        "native_modalities": ["temperature", "speed", "torque", "tool_wear"],
        "modality_channels": {"temperature": 2, "speed": 1, "torque": 1, "tool_wear": 1},
        "embed_dim": 32,
        "num_heads": 2,
        "num_fault_classes": 6,
        "dropout_p": 0.1
    }

@pytest.fixture
def ai4i_model(ai4i_config):
    return AI4IModel(**ai4i_config)

@pytest.fixture
def ai4i_sample_batch():
    B = 2
    seq_len = 1
    mod_values = {
        "temperature": torch.randn(B, seq_len, 2),
        "speed": torch.randn(B, seq_len, 1),
        "torque": torch.randn(B, seq_len, 1),
        "tool_wear": torch.randn(B, seq_len, 1),
    }
    mask = torch.ones(B, 4)
    return mod_values, mask


# --- Parameterized Tests ---

@pytest.mark.parametrize("model_fixture,batch_fixture,is_ai4i", [
    ("cmapss_model", "cmapss_sample_batch", False),
    ("ai4i_model", "ai4i_sample_batch", True)
])
def test_head_shapes(model_fixture, batch_fixture, is_ai4i, request):
    model = request.getfixturevalue(model_fixture)
    mod_values, mask = request.getfixturevalue(batch_fixture)
    
    model.eval()
    preds, _ = model(mod_values, mask)
    
    B = mod_values[list(mod_values.keys())[0]].shape[0]
    
    if is_ai4i:
        assert "rul_mean" not in preds
        assert "rul_variance" not in preds
        assert "anomaly_logit" in preds
        assert "fault_logits" in preds
        assert preds["anomaly_logit"].shape == (B, 1)
        assert preds["fault_logits"].shape == (B, 6) # 6 fault classes
    else:
        assert "rul_mean" in preds
        assert "rul_variance" in preds
        assert preds["rul_mean"].shape == (B, 1)
        assert preds["rul_variance"].shape == (B, 1)
        assert "anomaly_logit" not in preds
        assert "fault_logits" not in preds

@pytest.mark.parametrize("model_fixture,batch_fixture", [
    ("cmapss_model", "cmapss_sample_batch"),
    ("ai4i_model", "ai4i_sample_batch")
])
def test_masked_modality_zeroing(model_fixture, batch_fixture, request):
    model = request.getfixturevalue(model_fixture)
    mod_values, mask = request.getfixturevalue(batch_fixture)
    
    model.eval()
    
    # Mask out index 1 for sample 0
    mask[0, 1] = 0.0
    
    with torch.no_grad():
        preds, attn = model(mod_values, mask)
        
    assert hasattr(model.fusion, "_last_masked_out"), "Model must expose _last_masked_out for test verification."
    masked_out = model.fusion._last_masked_out
    
    seq_len = list(mod_values.values())[0].shape[1]
    
    # masked_out shape: [B * seq_len, num_modalities, embed_dim]
    # Sample 0 is indices 0 to seq_len-1. Modality 1.
    sample_0_masked_mod = masked_out[0:seq_len, 1, :]
    assert torch.all(sample_0_masked_mod == 0.0)

@pytest.mark.parametrize("model_fixture,batch_fixture", [
    ("cmapss_model", "cmapss_sample_batch"),
    ("ai4i_model", "ai4i_sample_batch")
])
def test_all_modalities_masked_no_nan(model_fixture, batch_fixture, request):
    model = request.getfixturevalue(model_fixture)
    mod_values, mask = request.getfixturevalue(batch_fixture)
    
    model.eval()
    
    # Mask ALL modalities for sample 0
    mask[0, :] = 0.0
    
    with torch.no_grad():
        preds, _ = model(mod_values, mask)
        
    if "rul_mean" in preds:
        assert not torch.isnan(preds["rul_mean"]).any()
        assert not torch.isnan(preds["rul_variance"]).any()
    if "anomaly_logit" in preds:
        assert not torch.isnan(preds["anomaly_logit"]).any()
    if "fault_logits" in preds:
        assert not torch.isnan(preds["fault_logits"]).any()

@pytest.mark.parametrize("model_fixture,batch_fixture", [
    ("cmapss_model", "cmapss_sample_batch"),
    ("ai4i_model", "ai4i_sample_batch")
])
def test_eval_determinism(model_fixture, batch_fixture, request):
    model = request.getfixturevalue(model_fixture)
    mod_values, mask = request.getfixturevalue(batch_fixture)
    
    model.eval()
    with torch.no_grad():
        preds1, _ = model(mod_values, mask)
        preds2, _ = model(mod_values, mask)
        
    if "rul_mean" in preds1:
        assert torch.allclose(preds1["rul_mean"], preds2["rul_mean"])
        assert torch.allclose(preds1["rul_variance"], preds2["rul_variance"])
    if "fault_logits" in preds1:
        assert torch.allclose(preds1["fault_logits"], preds2["fault_logits"])

@pytest.mark.parametrize("model_fixture,batch_fixture", [
    ("cmapss_model", "cmapss_sample_batch"),
    ("ai4i_model", "ai4i_sample_batch")
])
def test_train_non_determinism(model_fixture, batch_fixture, request):
    model = request.getfixturevalue(model_fixture)
    mod_values, mask = request.getfixturevalue(batch_fixture)
    
    model.train()
    with torch.no_grad():
        preds1, _ = model(mod_values, mask)
        preds2, _ = model(mod_values, mask)
        
    if "rul_mean" in preds1:
        assert not torch.allclose(preds1["rul_mean"], preds2["rul_mean"])
    if "fault_logits" in preds1:
        assert not torch.allclose(preds1["fault_logits"], preds2["fault_logits"])

@pytest.mark.parametrize("model_fixture,batch_fixture", [
    ("cmapss_model", "cmapss_sample_batch"),
    ("ai4i_model", "ai4i_sample_batch")
])
def test_train_fixed_seed_determinism(model_fixture, batch_fixture, request):
    model = request.getfixturevalue(model_fixture)
    mod_values, mask = request.getfixturevalue(batch_fixture)
    
    model.train()
    torch.manual_seed(42)
    with torch.no_grad():
        preds1, _ = model(mod_values, mask)
        
    torch.manual_seed(42)
    with torch.no_grad():
        preds2, _ = model(mod_values, mask)
        
    if "rul_mean" in preds1:
        assert torch.allclose(preds1["rul_mean"], preds2["rul_mean"])
        assert torch.allclose(preds1["rul_variance"], preds2["rul_variance"])
    if "fault_logits" in preds1:
        assert torch.allclose(preds1["fault_logits"], preds2["fault_logits"])
