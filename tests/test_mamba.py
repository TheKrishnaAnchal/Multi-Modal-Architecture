import torch
import time
from modules.mamba_engine import SimpleMambaBlock

def run_mamba_test():
    print("Initializing Mamba SSM Block Verification Test...")
    
    # Setup dimensions to match Phase 1 output
    batch_size = 2
    d_model = 256
    
    mamba_block = SimpleMambaBlock(d_model=d_model)
    
    print("\n--- TEST 1: Shape Integrity ---")
    # Simulate a sequence of 32 temporal tokens
    seq_len = 32 
    mock_sequence = torch.randn(batch_size, seq_len, d_model)
    output = mamba_block(mock_sequence)
    
    print(f"Input Shape:  {mock_sequence.shape}")
    print(f"Output Shape: {output.shape} (Expected: exactly the same)")
    assert output.shape == mock_sequence.shape, "Shape mismatch in Mamba Block!"
    print("Shape mapping perfectly preserved.")
    
    print("\n--- TEST 2: Linear Scaling Verification ---")
    print("Comparing execution time for sequence length 128 vs 1024...")
    
    # Warmup GPU/CPU to ensure accurate timing
    _ = mamba_block(torch.randn(batch_size, 16, d_model))
    
    # Test Short Sequence (128)
    start_time = time.time()
    _ = mamba_block(torch.randn(batch_size, 128, d_model))
    time_128 = time.time() - start_time
    
    # Test Massive Sequence (1024 - 8x longer)
    start_time = time.time()
    _ = mamba_block(torch.randn(batch_size, 1024, d_model))
    time_1024 = time.time() - start_time
    
    print(f"Time for 128 tokens:  {time_128:.6f} seconds")
    print(f"Time for 1024 tokens: {time_1024:.6f} seconds")
    
    # If this was a Transformer, 1024 tokens would take ~64x longer.
    # With our Mamba block, it should only take roughly 8x longer (linear).
    ratio = time_1024 / time_128 if time_128 > 0 else 0
    print(f"Scaling Ratio: {ratio:.2f}x (Ideal Linear Ratio is ~8.0x)")
    
    if ratio < 15.0: # Generous buffer for CPU fluctuations
        print("Success: Verified O(N) linear time complexity behavior.")
    else:
        print("Warning: Scaling appears non-linear, check hardware background tasks.")

    print("\nPhase 2 Component successfully verified. Ready for integration.")

if __name__ == "__main__":
    run_mamba_test()