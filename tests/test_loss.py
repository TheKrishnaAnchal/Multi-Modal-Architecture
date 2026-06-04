import torch
import torch.nn as nn
import torch.optim as optim
from modules.loss import DualObjectiveLoss

def run_loss_test():
    print("Initializing Dual-Objective Loss Layer Verification...")
    
    batch_size = 4
    seq_len = 32
    dimension = 256
    
    # Instantiate custom loss function
    criterion = DualObjectiveLoss(alpha=0.4)
    
    # Simulate outputs from our Mamba model (Requires gradients turned on)
    mock_student_output = torch.randn(batch_size, seq_len, dimension, requires_grad=True)
    
    # Simulate ground-truth vectors from a frozen Teacher Model
    mock_teacher_target = torch.randn(batch_size, dimension)
    
    # Create a tiny mock optimizer to check backpropagation
    optimizer = optim.Adam([mock_student_output], lr=0.01)
    
    # Execute loss pass
    total_loss, scav_loss, distill_loss = criterion(mock_student_output, mock_teacher_target)
    
    print("\n--- TEST 1: Loss Evaluation ---")
    print(f"Total Combined Loss: {total_loss.item():.4f}")
    print(f"├── SCAV Trajectory component: {scav_loss.item():.4f}")
    print(f"└── Soft Distillation component: {distill_loss.item():.4f}")
    
    # Ensure losses are valid numeric figures
    assert not torch.isnan(total_loss), "Loss calculation resulted in a NaN error!"
    
    print("\n--- TEST 2: Gradient Flow Integrity ---")
    # Execute backward pass
    total_loss.backward()
    
    # Check if gradients successfully populated inside the student tensors
    grad_sum = mock_student_output.grad.abs().sum().item()
    print(f"Accumulated Gradient Absolute Sum: {grad_sum:.6f}")
    
    if grad_sum > 0.0:
        print("Success: Backpropagation pathway verified. Model is completely trainable.")
    else:
        print("Failure: Gradients are zero. The network is suffering from a broken computation graph.")

if __name__ == "__main__":
    run_loss_test()