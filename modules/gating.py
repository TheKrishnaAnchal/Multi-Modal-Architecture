import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGatingMechanism(nn.Module):
    def __init__(self, visual_dim: int, audio_dim: int, joint_dim: int):
        """
        Adaptive Gating Mechanism (CMRF-Net inspired)
        Dynamically weights audio and visual tokens based on context and noise.
        """
        super(AdaptiveGatingMechanism, self).__init__()
        
        self.visual_projection = nn.Linear(visual_dim, joint_dim)
        self.audio_projection = nn.Linear(audio_dim, joint_dim)
        
        self.gate_network = nn.Sequential(
            nn.Linear(joint_dim * 2, joint_dim),
            nn.ReLU(),
            nn.Linear(joint_dim, 2), # Outputs: [Visual_Weight, Audio_Weight]
            nn.Softmax(dim=-1)       # Ensures weights add up to 1.0
        )
        
        self.ln_visual = nn.LayerNorm(joint_dim)
        self.ln_audio = nn.LayerNorm(joint_dim)

    def forward(self, visual_features: torch.Tensor, audio_features: torch.Tensor):
        v_proj = self.ln_visual(self.visual_projection(visual_features))
        a_proj = self.ln_audio(self.audio_projection(audio_features))
        
        combined_context = torch.cat([v_proj, a_proj], dim=-1)
        
        gating_weights = self.gate_network(combined_context)
        
        # THE FIX: Using [...] makes this safe for both 2D (Testing) and 3D (Pipeline) tensors
        v_weight = gating_weights[..., 0].unsqueeze(-1)
        a_weight = gating_weights[..., 1].unsqueeze(-1)
        
        fused_representation = (v_weight * v_proj) + (a_weight * a_proj)
        
        return fused_representation, (v_weight, a_weight)