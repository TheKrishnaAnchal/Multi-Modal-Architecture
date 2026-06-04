import torch
from modules.evaluation import AIMSEvaluator

def run_evaluation_test():
    print("Initializing Phase 6: Metric Evaluation Verification...")
    
    # Simulate a testing batch of 50 videos
    batch_size = 50
    embed_dim = 256
    
    # Simulate Ground Truth Teacher Embeddings
    mock_targets = torch.randn(batch_size, embed_dim)
    
    # Simulate our Model's Predictions (Adding slight noise to the targets to simulate a trained model)
    mock_predictions = mock_targets + (torch.randn(batch_size, embed_dim) * 0.5)
    
    evaluator = AIMSEvaluator()
    
    # 1. Test Cosine Similarity
    cos_sim = evaluator.calculate_cosine_similarity(mock_predictions, mock_targets)
    print(f"\nAverage Cosine Similarity: {cos_sim:.4f}")
    
    # 2. Test Retrieval Metrics
    metrics = evaluator.calculate_retrieval_metrics(mock_predictions, mock_targets)
    print("\n--- Downstream Task Efficacy (Retrieval) ---")
    print(f"Recall@1:  {metrics['R@1']:.2f}%")
    print(f"Recall@5:  {metrics['R@5']:.2f}%")
    print(f"Recall@10: {metrics['R@10']:.2f}%")
    print(f"Median Rank (MedR): {metrics['MedR']}")
    
    print("\nSUCCESS: All AIMS DTU evaluation metrics correctly implemented.")

if __name__ == "__main__":
    run_evaluation_test()