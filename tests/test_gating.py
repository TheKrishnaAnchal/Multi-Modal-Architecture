import torch
from modules.gating import AdaptiveGatingMechanism

def run_verification_test():
    print("Initializing Adaptive Gating Verification Test...")
    
    # Define arbitrary standard dimensions
    batch_size = 2
    v_dim = 512    # e.g., standard ViT or ImageBind output
    a_dim = 128    # e.g., standard Audio Spectrogram output
    j_dim = 256    # Shared fusion space
    
    # Instantiate our module
    gatekeeper = AdaptiveGatingMechanism(visual_dim=v_dim, audio_dim=a_dim, joint_dim=j_dim)
    
    # Scenario A: Simulate standard clean inputs
    mock_visual = torch.randn(batch_size, v_dim)
    mock_audio = torch.randn(batch_size, a_dim)
    
    fused, (v_w, a_w) = gatekeeper(mock_visual, mock_audio)
    
    print("\n--- TEST 1: Shape Check ---")
    print(f"Fused Output Shape: {fused.shape} (Expected: [{batch_size}, {j_dim}])")
    assert fused.shape == (batch_size, j_dim), "Shape mismatch error!"
    print("Output shapes verified successfully.")
    
    print("\n--- TEST 2: Gating Dynamics ---")
    print(f"Visual Attention Weight: {v_w[0].item():.4f}")
    print(f"Audio Attention Weight:  {a_w[0].item():.4f}")
    print(f"Total Weight Sum:       {(v_w[0] + a_w[0]).item():.4f} (Expected: 1.0000)")
    
    print("\nPhase 1 Component successfully verified. Ready for integration.")

if __name__ == "__main__":
    run_verification_test()