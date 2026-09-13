import torch
import numpy as np

try:
    from captum.attr import GradientShap
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

class MESHExplainer:
    def __init__(self, model, native_modalities):
        self.model = model
        self.native_modalities = native_modalities
        self.device = next(model.parameters()).device
        self.model.eval()

    def get_attention_weights(self, modality_values, modality_mask):
        """
        Runs a forward pass and returns the explicit mask-aware fusion attention weights.
        """
        with torch.no_grad():
            _, attn_weights = self.model(modality_values, modality_mask)
        
        # attn_weights shape: [B, seq_len, num_heads, num_modalities, num_modalities]
        # Zero out the rows corresponding to missing modalities (they are uniformly random before maskout)
        # modality_mask shape: [B, num_modalities] -> [B, 1, 1, num_modalities, 1]
        mask_expanded = modality_mask.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
        attn_weights = attn_weights * mask_expanded
        return attn_weights

    def get_gradientshap_attribution(self, modality_values, modality_mask, target_head="rul", target_class=None):
        """
        Computes GradientSHAP attributions for the inputs.
        CAVEAT: On an undertrained model (or one trained purely on random noise), 
        these attributions will NOT reflect meaningful feature importance.
        """
        if not CAPTUM_AVAILABLE:
             raise ImportError("Captum is required for GradientSHAP. Please `pip install captum`.")
             
        input_tensors = tuple(modality_values[m] for m in self.native_modalities)
        baseline_tensors = tuple(torch.zeros_like(t) for t in input_tensors)

        def forward_wrapper(*inputs):
            mod_dict = {m: inp for m, inp in zip(self.native_modalities, inputs)}
            
            # Captum expands the batch dimension of inputs (e.g. from 2 to 10 for num_samples=5).
            # The mask must be expanded to match the new batch size.
            B_actual = inputs[0].shape[0]
            B_orig = modality_mask.shape[0]
            
            if B_actual != B_orig:
                repeats = B_actual // B_orig
                current_mask = modality_mask.repeat_interleave(repeats, dim=0)
            else:
                current_mask = modality_mask
                
            preds, _ = self.model(mod_dict, current_mask)
            
            if target_head == "rul":
                return preds["rul_mean"]
            elif target_head == "anomaly":
                return preds["anomaly_logit"]
            elif target_head == "fault":
                return preds["fault_logits"]
            else:
                raise ValueError("Unknown target_head")

        gs = GradientShap(forward_wrapper)
        
        # Calculate attributions. If target_head is fault, target_class must be provided.
        # For scalar outputs like rul and anomaly, target should be 0 or None depending on shape.
        if target_head == "fault":
             attributions = gs.attribute(input_tensors, baselines=baseline_tensors, target=target_class)
        else:
             # If output is [batch, 1], captum expects target=0.
             attributions = gs.attribute(input_tensors, baselines=baseline_tensors, target=0)
             
        # Return as a dictionary mapping modality name to its attribution tensor
        return {m: attr for m, attr in zip(self.native_modalities, attributions)}

if __name__ == "__main__":
    from ml.data.mock_canonical_batch import generate_mock_canonical_batch
    
    print("=== MESH Explainability Diagnostics ===")
    print("CAVEAT: The model currently evaluated is trained on MOCK uniform noise.")
    print("The GradientSHAP outputs below DO NOT imply real feature importance.")
    print("This run exclusively verifies PIPELINE CORRECTNESS.\n")
    
    if not CAPTUM_AVAILABLE:
        print("Captum not found. Please run `pip install captum`.")
        exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    from ml.models.mesh_model import MESHModel
    checkpoint = torch.load("checkpoints/model_best.pt", map_location=device, weights_only=False)
    config = checkpoint['model_config']
    
    native_modalities = ["temperature", "tool_wear", "rotational_speed", "torque"]
    model = MESHModel(
        native_modalities=native_modalities,
        cnn_in_channels_map={m: 1 for m in native_modalities},
        encoder_config=config['encoder'],
        fusion_config=config['fusion'],
        temporal_config=config['temporal'],
        heads_config=config['heads'],
        dropout_p=0.0
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    
    explainer = MESHExplainer(model, native_modalities)
    
    # Generate 1 small batch
    batch = generate_mock_canonical_batch(batch_size=2, window_size=20)
    mod_values = {k: v.to(device) for k, v in batch.modality_values.items()}
    # Let's forcefully mask out modality 1 (tool_wear) for BOTH samples to verify attention is 0
    mask = batch.modality_mask.to(device)
    mask[:, 1] = 0.0 
    
    print("1. ATTENTION WEIGHTS (Mask-Aware Fusion)")
    attn_weights = explainer.get_attention_weights(mod_values, mask)
    print(f"Shape: {attn_weights.shape}")
    print(f"Attention Weights for Sample 0:\n{attn_weights[0]}")
    print(f"Is tool_wear (index 1) query exactly 0 in attention? {torch.all(attn_weights[:, :, :, 1, :] == 0).item()}")
    print(f"Is tool_wear (index 1) key exactly 0 in attention? {torch.all(attn_weights[:, :, :, :, 1] == 0).item()}")
    
    print("\n2. GRADIENT SHAP ATTRIBUTIONS (RUL Head)")
    attributions = explainer.get_gradientshap_attribution(mod_values, mask, target_head="rul")
    for mod, attr in attributions.items():
        print(f"  {mod} Attribution Shape: {attr.shape} | Mean absolute attribution: {torch.abs(attr).mean().item():.6f}")
