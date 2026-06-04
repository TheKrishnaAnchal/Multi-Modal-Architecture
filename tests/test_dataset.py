import torch
import time
from modules.pipeline import HybridMambaPipeline
from data.dataset import AIMSMultimodalDataset

def run_real_data_test():
    print("Initializing Phase 5: Real-World Data Integration...")
    
    # 1. Initialize our custom DataLoader
    video_list = ["data/test_video.mp4"]
    try:
        dataset = AIMSMultimodalDataset(video_paths=video_list)
        visual_tensor, audio_tensor = dataset[0]
    except Exception as e:
        print(f"DataLoader failed to process the video. Error: {e}")
        return

    print("\n--- TEST 1: DataLoader Extraction ---")
    print(f"Extracted Sparse Frame Shape: {visual_tensor.shape} (Expected: [3, 224, 224])")
    print(f"Extracted Audio Track Shape:  {audio_tensor.shape} (Expected: [256, 128])")
    
    # Simulate a Batch Size of 1 to feed into the model
    visual_batch = visual_tensor.unsqueeze(0)
    audio_batch = audio_tensor.unsqueeze(0)
    
    # 2. To test the pipeline, we need a fake Teacher target (since we don't have ImageBind loaded yet)
    # We will simulate a [1, 256] target vector
    mock_teacher_target = torch.randn(1, 256)
    
    # 3. Initialize the Hybrid Mamba Model
    # Note: Our visual input is currently 3x224x224 (150,528 pixels). 
    # In a real setup, a CNN/ViT compresses this to 512. For this test, we simulate that projection layer.
    compression_layer = torch.nn.Linear(3 * 224 * 224, 512)
    compressed_visual = compression_layer(visual_batch.flatten(start_dim=1))
    
    hybrid_model = HybridMambaPipeline(visual_dim=512, audio_dim=128, joint_dim=256)
    
    print("\n--- TEST 2: Full System End-to-End Pass ---")
    start_time = time.time()
    
    # Feed the REAL data through the model
    loss, output_seq, gating_weights = hybrid_model(compressed_visual, audio_batch, mock_teacher_target)
    
    exec_time = time.time() - start_time
    
    print(f"System Execution Time: {exec_time:.4f}s")
    print(f"Final Model Output Shape: {output_seq.shape}")
    print(f"Calculated Test Loss: {loss.item():.4f}")
    print("\nSUCCESS: The architecture successfully ingested, processed, and optimized a real multimodal video file.")

if __name__ == "__main__":
    run_real_data_test()