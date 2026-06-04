import torch
import torch.nn.functional as F

class AIMSEvaluator:
    def __init__(self):
        """
        Calculates the specific evaluation metrics required by the AIMS DTU 2026 Mandate:
        Cosine Similarity, Recall@K, and Median Rank.
        """
        pass

    def calculate_cosine_similarity(self, predicted_embeddings: torch.Tensor, target_embeddings: torch.Tensor) -> float:
        """
        Measures the angular distance between the predicted audio-visual embedding 
        and the ground-truth video embedding.
        """
        # Normalize vectors
        pred_norm = F.normalize(predicted_embeddings, p=2, dim=-1)
        target_norm = F.normalize(target_embeddings, p=2, dim=-1)
        
        # Calculate Cosine Similarity
        cos_sim = torch.sum(pred_norm * target_norm, dim=-1).mean().item()
        return cos_sim

    def calculate_retrieval_metrics(self, predicted_embeddings: torch.Tensor, target_embeddings: torch.Tensor):
        """
        Calculates cross-modal retrieval metrics: Recall@1, Recall@5, Recall@10, and Median Rank.
        This tests if the approximated embedding is practically viable for semantic search.
        """
        batch_size = predicted_embeddings.size(0)
        
        # Normalize vectors for distance calculation
        pred_norm = F.normalize(predicted_embeddings, p=2, dim=-1)
        target_norm = F.normalize(target_embeddings, p=2, dim=-1)
        
        # Calculate the similarity matrix (Every prediction compared to every target)
        similarity_matrix = torch.matmul(pred_norm, target_norm.T) # Shape: [Batch, Batch]
        
        # Sort the matrix to find the rank of the correct target for each prediction
        # The correct target is on the diagonal of the matrix (e.g., Pred 0 matches Target 0)
        sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
        
        ranks = []
        for i in range(batch_size):
            # Find where the true matching index (i) is located in the sorted predictions
            rank = (sorted_indices[i] == i).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
            
        ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
        
        # Calculate Recall@K
        r1 = (ranks_tensor <= 1).float().mean().item() * 100
        r5 = (ranks_tensor <= 5).float().mean().item() * 100
        r10 = (ranks_tensor <= 10).float().mean().item() * 100
        
        # Calculate Median Rank
        medr = torch.median(ranks_tensor).item()
        
        return {
            "R@1": r1,
            "R@5": r5,
            "R@10": r10,
            "MedR": medr
        }