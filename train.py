import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from modules.pipeline import HybridMambaPipeline
from data.dataset import AIMSMultimodalDataset

def train_midnight_sprint():
    print("Initializing Midnight Crunch Run | Hyper-Optimized for Cosine Similarity...")
    
    # 1. Aggressive Configuration
    EPOCHS = 20 
    BATCH_SIZE = 8 # Push this as high as your GPU allows before crashing (8, 16, or 32)
    LEARNING_RATE = 3e-4 # Tripled from earlier to force faster learning
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware locked: {DEVICE}")
    
    # 2. Data Loading (Assuming you have your data in the folder)
    data_dir = "data/"
    video_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mp4') or f.endswith('.avi')]
    
    if len(video_paths) == 0:
        print("CRITICAL: Drop your video clips into the 'data/' folder immediately.")
        return
        
    print(f"Targeting {len(video_paths)} clips for the sprint.")
    dataset = AIMSMultimodalDataset(video_paths)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Initialize Architecture with the Alpha Shift (alpha=0.15 inside pipeline.py)
    model = HybridMambaPipeline(visual_dim=512, audio_dim=128, joint_dim=256).to(DEVICE)
    visual_projector = torch.nn.Linear(3 * 224 * 224, 512).to(DEVICE)
    
    # 4. The Accelerator Optimizer
    optimizer = optim.AdamW(list(model.parameters()) + list(visual_projector.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    # This dynamically curves the learning rate to squeeze out extra accuracy
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 5. Sprint Training Loop
    model.train()
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        running_cos_sim = 0.0
        
        for batch_idx, (visual_frames, audio_seqs) in enumerate(dataloader):
            visual_frames, audio_seqs = visual_frames.to(DEVICE), audio_seqs.to(DEVICE)
            optimizer.zero_grad()
            
            visual_features = visual_projector(visual_frames.flatten(start_dim=1))
            
            # Simulated Teacher (Replace with real targets if you pre-extracted them)
            mock_teacher_target = torch.randn(visual_frames.size(0), 256).to(DEVICE)
            
            # Forward Pass
            loss, output_seq, _ = model(visual_features, audio_seqs, mock_teacher_target)
            
            # Calculate Live Cosine Similarity to monitor the spike
            output_global = F.normalize(output_seq.mean(dim=1), p=2, dim=-1)
            target_norm = F.normalize(mock_teacher_target, p=2, dim=-1)
            batch_cos_sim = torch.sum(output_global * target_norm, dim=-1).mean().item()
            running_cos_sim += batch_cos_sim
            
            # Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevents gradient explosions
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        avg_cos_sim = running_cos_sim / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Live Cosine Sim: {avg_cos_sim:.4f}")

    print(f"\nSprint Complete in {(time.time() - start_time) / 60:.2f} minutes.")
    torch.save(model.state_dict(), "mamba_midnight_weights.pth")

if __name__ == "__main__":
    train_midnight_sprint()