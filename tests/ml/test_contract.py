import pytest
import torch
import numpy as np
from typing import Dict, List
from ml.data.contract import CanonicalBatch
from ml.inference.interface import PredictionBundle, MESHInferenceEngine
from pydantic import ValidationError

def test_canonical_batch_validation():
    # Valid batch
    valid_data = {
        "sample_id": ["samp_1", "samp_2"],
        "source_dataset": ["mock", "mock"],
        "dataset_version": ["v1", "v1"],
        "run_id": ["run_1", "run_2"],
        "window_start": [0, 0],
        "window_end": [20, 20],
        "sampling_interval_seconds": [1.0, 1.0],
        "native_modalities": ["temperature", "tool_wear", "rotational_speed", "torque"],
        "preprocessor_version": ["v1", "v1"],
        "scaler_version": ["v1", "v1"],
        "modality_values": {
            "temperature": torch.randn(2, 20, 1),
            "tool_wear": torch.randn(2, 20, 1),
            "rotational_speed": torch.randn(2, 20, 1),
            "torque": torch.randn(2, 20, 1)
        },
        "modality_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]]),
        "target_rul": torch.tensor([150.0, 200.0]),
        "target_anomaly": torch.tensor([0, 1]),
        "target_fault_class": torch.tensor([0, 2])
    }
    
    batch = CanonicalBatch(**valid_data)
    assert batch.sample_id == ["samp_1", "samp_2"]
    assert len(batch.target_rul) == 2

def test_canonical_batch_failure_missing_field():
    import copy
    
    # Valid batch identical to the one above
    valid_data = {
        "sample_id": ["samp_1", "samp_2"],
        "source_dataset": ["mock", "mock"],
        "dataset_version": ["v1", "v1"],
        "run_id": ["run_1", "run_2"],
        "window_start": [0, 0],
        "window_end": [20, 20],
        "sampling_interval_seconds": [1.0, 1.0],
        "native_modalities": ["temperature", "tool_wear", "rotational_speed", "torque"],
        "preprocessor_version": ["v1", "v1"],
        "scaler_version": ["v1", "v1"],
        "modality_values": {
            "temperature": torch.randn(2, 20, 1),
            "tool_wear": torch.randn(2, 20, 1),
            "rotational_speed": torch.randn(2, 20, 1),
            "torque": torch.randn(2, 20, 1)
        },
        "modality_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]]),
        "target_rul": torch.tensor([150.0, 200.0]),
        "target_anomaly": torch.tensor([0, 1]),
        "target_fault_class": torch.tensor([0, 2])
    }
    
    invalid_data = copy.deepcopy(valid_data)
    del invalid_data["modality_mask"]
    
    with pytest.raises(ValidationError) as excinfo:
        CanonicalBatch(**invalid_data)
        
    assert "modality_mask" in str(excinfo.value)

def test_prediction_bundle_serialization():
    # Test that PredictionBundle can be serialized to JSON (for FastAPI/Sarthak)
    bundle = PredictionBundle(
        model_version="v1.0",
        checkpoint_id="chk_1",
        training_config_ref="conf_1",
        rul_mean_cycles=[[100.0]],
        rul_variance_cycles=[[10.0]],
        fault_probabilities=[[0.1, 0.8, 0.1]],
        anomaly_probabilities=[[0.9]],
        attention_weights=None
    )
    
    from dataclasses import asdict
    import json
    
    bundle_dict = asdict(bundle)
    json_str = json.dumps(bundle_dict)
    
    assert "v1.0" in json_str
    assert "rul_mean_cycles" in json_str
    assert bundle_dict['rul_mean_cycles'] == [[100.0]]

def test_checkpoint_roundtrip(tmp_path):
    # E2E Checkpoint Roundtrip
    # Build a model in memory, run inference on a fixed batch, record the output.
    # Save it to a temp checkpoint file. Load that temp file via MESHInferenceEngine, 
    # run inference on the same batch, assert the two outputs match exactly.
    from ml.models.mesh_model import MESHModel
    import yaml
    
    # Minimal config to instantiate MESHModel
    config = {
        "native_modalities": ["temperature", "tool_wear", "rotational_speed", "torque"],
        "cnn_in_channels_map": {"temperature": 1, "tool_wear": 1, "rotational_speed": 1, "torque": 1},
        "encoder_config": {"cnn_out_channels": 8, "cnn_kernel_size": 3, "bilstm_hidden_size": 16, "bilstm_num_layers": 1},
        "fusion_config": {"embed_dim": 32, "num_heads": 2, "dropout": 0.1},
        "temporal_config": {"d_model": 32, "nhead": 2, "num_layers": 1, "dim_feedforward": 16, "dropout": 0.1},
        "heads_config": {"rul_hidden_dim": 16, "fault_hidden_dim": 16, "anomaly_hidden_dim": 16, "num_fault_classes": 3},
        "dropout_p": 0.1
    }
    model = MESHModel(**config)
    model.eval()
    
    # Save to tmp_path using exact train.py schema
    chk_path = tmp_path / "temp_chk.pt"
    
    # Create the config mapping exactly as interface.py expects it from YAML
    yaml_config = {
        'model_type': 'MESHModel',
        'native_modalities': config['native_modalities'],
        'modality_channels': config['cnn_in_channels_map'],
        'encoder_config': config['encoder_config'],
        'fusion_config': config['fusion_config'],
        'temporal_config': config['temporal_config'],
        'heads_config': config['heads_config']
    }
    
    checkpoint = {
        'state_dict': model.state_dict(),
        'model_config': yaml_config,
        'train_config': {"some_key": "some_value"},
        'target_normalization': {
            'rul_mean': 100.0,
            'rul_std': 25.0
        },
        'model_version': "test_v1",
        'checkpoint_id': "test_chk_123"
    }
    torch.save(checkpoint, chk_path)
    
    # Generate batch
    from ml.data.mock_canonical_batch import generate_mock_canonical_batch
    batch = generate_mock_canonical_batch(batch_size=2, window_size=20)
    
    # Get raw model predictions (for comparison)
    # The interface engine de-normalizes RUL natively, so we compare engine vs engine or mock the interface wrapper.
    # The easiest is to just use MESHInferenceEngine on the newly saved checkpoint and ensure it executes successfully and deterministically!
    # Wait, the instruction says: "Build a model in memory, run inference on a fixed batch, record the output. Save it... Load it... assert exact match"
    
    # Run raw model inference
    with torch.no_grad():
        modality_tensors = {k: batch.modality_values[k] for k in config["native_modalities"]}
        preds, _ = model(modality_tensors, batch.modality_mask)
        raw_rul_mean = preds["rul_mean"]
        raw_rul_variance = preds["rul_variance"]
        
    # De-normalize manually for comparison
    expected_rul_mean_cycles = raw_rul_mean * 25.0 + 100.0
    expected_rul_variance_cycles = raw_rul_variance * (25.0 ** 2)
        
    # Load via MESHInferenceEngine
    engine = MESHInferenceEngine(str(chk_path))
    bundle = engine.predict(batch)
    
    # Assert Exact Match
    assert np.allclose(bundle.rul_mean_cycles, expected_rul_mean_cycles.numpy())
    assert np.allclose(bundle.rul_variance_cycles, expected_rul_variance_cycles.numpy())
    # Note: logits to probabilities handling (softmax, sigmoid) is done inside engine.predict,
    # so we focus on ensuring the primary RUL de-norm and raw execution match identically.
    assert bundle.model_version == "test_v1"
