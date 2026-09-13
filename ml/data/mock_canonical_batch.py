import torch
from typing import List
from ml.data.contract import CanonicalBatch

def generate_mock_canonical_batch(
    batch_size: int = 16,
    window_size: int = 20,
    native_modalities: List[str] = None
) -> CanonicalBatch:
    """
    TEST FIXTURE: Generates random tensors matching the CanonicalBatch contract.
    This contains no real data logic and is purely for unblocking ML development
    while the data pipeline is being built by Naman.
    """
    if native_modalities is None:
        native_modalities = ["temperature", "tool_wear", "rotational_speed", "torque"]
        
    num_modalities = len(native_modalities)
    
    # 1. Generate mocked dictionary of modality values
    modality_values = {}
    for mod in native_modalities:
        # Assuming 1 feature dimension for simplicity in the mock
        if mod == "tool_wear":
            # Monotonically increasing tool wear simulation, scaled to roughly [-1, 1]
            base = torch.linspace(-1.0, 1.0, window_size).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)
            noise = (torch.rand(batch_size, window_size, 1) - 0.5) * 0.1
            modality_values[mod] = base + noise
        else:
            modality_values[mod] = torch.randn(batch_size, window_size, 1)
        
    # 2. Generate modality mask (randomly drop some modalities for robustness testing)
    # Mask shape: [batch_size, num_modalities]
    # For a simple mock, we'll make them all 1s, but randomly zero out a few
    modality_mask = torch.ones(batch_size, num_modalities)
    # E.g. drop the first modality for the first sample
    if batch_size > 0:
        modality_mask[0, 0] = 0.0

    # 3. Generate mocked targets
    target_rul = torch.randint(10, 200, (batch_size, 1)).float()
    target_fault_class = torch.randint(0, 3, (batch_size,))
    target_degradation = torch.rand(batch_size, 1)
    
    return CanonicalBatch(
        sample_id=[f"sample_{i}" for i in range(batch_size)],
        source_dataset=["mock_dataset"] * batch_size,
        dataset_version=["v0.1.mock"] * batch_size,
        run_id=[f"run_{i%5}" for i in range(batch_size)],
        machine_id=[f"machine_{i%2}" for i in range(batch_size)],
        window_start=[float(i) for i in range(batch_size)],
        window_end=[float(i + window_size) for i in range(batch_size)],
        sampling_interval_seconds=[1.0] * batch_size,
        native_modalities=native_modalities,
        preprocessor_version=["v0.1.mock"] * batch_size,
        scaler_version=["v0.1.mock"] * batch_size,
        modality_values=modality_values,
        modality_mask=modality_mask,
        target_rul=target_rul,
        target_fault_class=target_fault_class,
        target_degradation=target_degradation
    )
