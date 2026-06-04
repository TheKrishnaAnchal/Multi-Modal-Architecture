import torch
import torch.nn as nn
from modules.gating import AdaptiveGatingMechanism
from modules.mamba_engine import SimpleMambaBlock
from modules.loss import DualObjectiveLoss

class NaiveBaselineModel(nn.Module):
    def __init__(self, visual_dim: int, audio_dim: int, joint_dim: int):
        """
        The traditional baseline model. 
        It forces audio and video together without dynamic gating and 
        squashes the sequence into a single point before alignment.
        """
        super(NaiveBaselineModel, self).__init__()
        self.v_proj = nn.Linear(visual_dim, joint_dim)
        self.a_proj = nn.Linear(audio_dim, joint_dim)
        
        # Standard Multi-Layer Perceptron (MLP) for fusion instead of Mamba
        self.fusion_mlp = nn.Sequential(
            nn.Linear(joint_dim, joint_dim * 2),
            nn.ReLU(),
            nn.Linear(joint_dim * 2, joint_dim)
        )
        self.loss_fn = nn.MSELoss() # Standard rigid L2 loss (Causes collapse)

    def forward(self, visual_features: torch.Tensor, audio_seq: torch.Tensor, teacher_global: torch.Tensor):
        # 1. Standard projection
        v_proj = self.v_proj(visual_features)
        
        # 2. Temporal Pooling (This is the fatal flaw: it squashes the timeline)
        a_squashed = self.a_proj(audio_seq).mean(dim=1) 
        
        # 3. Simple addition fusion
        fused = self.fusion_mlp(v_proj + a_squashed)
        
        # 4. Standard rigid loss
        loss = self.loss_fn(fused, teacher_global)
        return loss, fused

class HybridMambaPipeline(nn.Module):
    def __init__(self, visual_dim: int, audio_dim: int, joint_dim: int):
        """
        The AIMS DTU 2026 Master Architecture.
        Integrates Adaptive Gating, Mamba Sequence Engine, and Dual-Objective Loss.
        """
        super(HybridMambaPipeline, self).__init__()
        
        # Phase 1: Adaptive Gatekeeper
        self.gatekeeper = AdaptiveGatingMechanism(visual_dim, audio_dim, joint_dim)
        
        # Phase 2: High-Speed Mamba Engine
        self.mamba_core = SimpleMambaBlock(d_model=joint_dim)
        
        # Phase 3: Dual-Objective Optimizer
        self.criterion = DualObjectiveLoss(alpha=0.6)

    def forward(self, visual_features: torch.Tensor, audio_seq: torch.Tensor, teacher_global: torch.Tensor):
        """
        Input dimensions:
        visual_features: [Batch, V_Dim] (The sparse frame)
        audio_seq: [Batch, Seq_Len, A_Dim] (The continuous audio track)
        teacher_global: [Batch, Joint_Dim] (The frozen ground truth)
        """
        batch_size, seq_len, _ = audio_seq.shape
        
        # Step 1: Expand the sparse visual frame to match the audio sequence length
        visual_expanded = visual_features.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Step 2: Apply adaptive gating at every timestep
        fused_seq, gating_weights = self.gatekeeper(visual_expanded, audio_seq)
        
        # Step 3: Process the sequence chronologically via Mamba
        processed_seq = self.mamba_core(fused_seq)
        
        # Step 4: Calculate advanced loss (No squashing!)
        total_loss, loss_scav, loss_distill = self.criterion(processed_seq, teacher_global)
        
        return total_loss, processed_seq, gating_weights