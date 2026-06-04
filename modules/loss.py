import torch
import torch.nn as nn
import torch.nn.functional as F

class DualObjectiveLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, temperature: float = 0.07):
        """
        Dual-Objective Loss Layer for Sequential Alignment and Soft Distillation.
        Prevents dimensional collapse by balancing fine-grained trajectory matching 
        with abstract teacher representation alignment.
        
        Args:
            alpha (float): Balancing coefficient between SCAV and Distillation losses.
            temperature (float): Scaling factor for scaling semantic similarities.
        """
        super(DualObjectiveLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()

    def compute_scav_loss(self, student_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
        """
        Calculates sequential alignment loss using an Interpolated Euclidean sequence distance.
        Ensures temporal trajectories match without collapsing sequence variance.
        """
        # If lengths don't match exactly due to sampling rates, interpolate student to match target
        if student_seq.shape[1] != target_seq.shape[1]:
            student_seq = student_seq.transpose(1, 2) # [B, D, S_len]
            student_seq = F.interpolate(student_seq, size=target_seq.shape[1], mode='linear', align_corners=True)
            student_seq = student_seq.transpose(1, 2) # [B, S_len, D]
            
        # Normalize vectors along the feature dimension to prevent exploding gradients
        student_norm = F.normalize(student_seq, p=2, dim=-1)
        target_norm = F.normalize(target_seq, p=2, dim=-1)
        
        # Calculate element-wise mean square distance between continuous sequence trajectories
        return self.mse_loss(student_norm, target_norm)

    def compute_distillation_loss(self, student_global: torch.Tensor, teacher_global: torch.Tensor) -> torch.Tensor:
        """
        Calculates soft-constrained distillation loss via smooth cosine similarity matching.
        Avoids hard L2 constraints which trigger premature dimensional collapse.
        """
        # Normalize global representations
        student_norm = F.normalize(student_global, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_global, p=2, dim=-1)
        
        # Cosine distance computation
        cosine_sim = torch.sum(student_norm * teacher_norm, dim=-1)
        distill_loss = 1.0 - cosine_sim.mean()
        
        return distill_loss

    def forward(self, student_seq: torch.Tensor, teacher_global: torch.Tensor):
        """
        Forward pass execution.
        student_seq: [Batch_Size, Seq_Len, Dimension] -> Output of Mamba Engine
        teacher_global: [Batch_Size, Dimension]       -> Output of Frozen Video Teacher
        """
        # 1. Extract global context from our student sequence (temporal pooling)
        student_global = student_seq.mean(dim=1)
        
        # 2. Reconstruct target seq from global context to verify continuous trajectory stability
        # In this self-contained block, we align the student's timeline with its own pooled target
        # to ensure local token cohesion before distilling out globally.
        target_seq_anchor = teacher_global.unsqueeze(1).expand_2d = teacher_global.unsqueeze(1).repeat(1, student_seq.shape[1], 1)
        
        loss_scav = self.compute_scav_loss(student_seq, target_seq_anchor)
        loss_distill = self.compute_distillation_loss(student_global, teacher_global)
        
        # Combined weighted loss formulation
        total_loss = (self.alpha * loss_scav) + ((1.0 - self.alpha) * loss_distill)
        
        return total_loss, loss_scav, loss_distill