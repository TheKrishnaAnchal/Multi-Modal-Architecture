import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        """
        Simplified State Space Model (SSM) Block
        Processes sequences in linear time O(N) by bypassing dense attention mechanisms.
        
        Args:
            d_model (int): Dimension of the input features (e.g., 256 from Phase 1)
            d_state (int): Size of the hidden state matrix
            expand (int): Expansion factor for internal processing width
        """
        super(SimpleMambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state  # Save d_state explicitly to prevent splitting bugs
        d_inner = int(expand * d_model)
        
        # 1. Input Projections
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        
        # 2. 1D Causal Convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_inner, 
            out_channels=d_inner, 
            kernel_size=4, 
            padding=3, 
            groups=d_inner
        )
        
        # 3. State Space Parameters (1 column for delta, d_state for B, d_state for C)
        self.x_proj = nn.Linear(d_inner, 1 + 2 * d_state)
        self.dt_proj = nn.Linear(1, d_inner)
        
        # 4. Output Projection
        self.out_proj = nn.Linear(d_inner, d_model)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        Forward pass execution.
        Input shape: [Batch_Size, Sequence_Length, D_Model]
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # Normalize input
        x = self.norm(x)
        
        # Step 1: Project input and split into two branches (Activation & Gating)
        x_proj = self.in_proj(x)
        x_hidden, x_gate = x_proj.chunk(2, dim=-1)
        
        # Step 2: Causal Convolution
        x_hidden = x_hidden.transpose(1, 2)
        x_hidden = self.conv1d(x_hidden)[:, :, :seq_len] # Truncate padding to maintain causality
        x_hidden = x_hidden.transpose(1, 2)
        
        # Apply SiLU activation
        x_hidden = F.silu(x_hidden)
        
        # Step 3: Explicit Data-Dependent SSM Gating
        ssm_params = self.x_proj(x_hidden)
        
        # Explicitly split into 1, 16, and 16 columns. This perfectly equals 33 columns total.
        delta, B, C = torch.split(ssm_params, [1, self.d_state, self.d_state], dim=-1)
        
        # Discretize and apply
        delta = F.softplus(delta)
        gating_weight = torch.sigmoid(self.dt_proj(delta))
        
        # Modulate the hidden state with the gate
        x_fused = x_hidden * gating_weight
        
        # Step 4: Re-multiply with the original residual gate and project out
        x_out = x_fused * F.silu(x_gate)
        output = self.out_proj(x_out)
        
        return output + residual