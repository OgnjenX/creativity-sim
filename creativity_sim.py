"""
Creativity Simulation using PyTorch

This script simulates creativity by combining random latent vectors with varying
degrees of exploration (controlled by alpha) and measuring the novelty and coherence
of the resulting combinations.
"""

import torch
import matplotlib.pyplot as plt


def reorganize(x_i, x_j, alpha, noise):
    """
    Combine two input vectors with weighted averaging and scaled noise.
    
    Args:
        x_i: First input vector (tensor)
        x_j: Second input vector (tensor)
        alpha: Weighting factor for combination (float between 0 and 1)
        noise: Random noise vector (tensor)
    
    Returns:
        Combined vector with noise scaled by (1-alpha)
    """
    # Weighted combination of input vectors
    combined = alpha * x_i + (1 - alpha) * x_j
    # Add noise scaled by (1-alpha) for exploration
    result = combined + (1 - alpha) * noise
    return result


def compute_novelty(vector, memory):
    """
    Compute novelty as the minimum distance to all vectors in memory.
    
    Args:
        vector: Query vector to evaluate (tensor of shape [dim])
        memory: Collection of memory vectors (tensor of shape [n_memory, dim])
    
    Returns:
        Novelty score (minimum Euclidean distance to memory)
    """
    # Compute Euclidean distances to all memory vectors
    distances = torch.norm(memory - vector.unsqueeze(0), dim=1)
    # Novelty is the minimum distance (how different from everything we've seen)
    novelty = torch.min(distances)
    return novelty


def compute_coherence(vector, x_i, x_j):
    """
    Compute coherence as the average cosine similarity to input vectors.
    
    Args:
        vector: Output vector to evaluate (tensor)
        x_i: First input vector (tensor)
        x_j: Second input vector (tensor)
    
    Returns:
        Coherence score (average cosine similarity to inputs)
    """
    # Cosine similarity with first input
    cos_sim_i = torch.nn.functional.cosine_similarity(
        vector.unsqueeze(0), x_i.unsqueeze(0)
    )[0]
    # Cosine similarity with second input
    cos_sim_j = torch.nn.functional.cosine_similarity(
        vector.unsqueeze(0), x_j.unsqueeze(0)
    )[0]
    # Coherence is the average similarity (how related to inputs)
    coherence = (cos_sim_i + cos_sim_j) / 2.0
    return coherence


def main():
    """
    Main simulation loop for creativity exploration.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize parameters
    dim = 16  # Dimension of latent vectors
    n_initial_memory = 100  # Initial number of memory vectors
    n_steps = 100  # Number of simulation steps
    
    # Initialize memory with 100 random latent vectors (dimension 16)
    memory = torch.randn(n_initial_memory, dim)
    
    # Storage for creativity scores
    creativity_scores = []
    
    # Main simulation loop
    for step in range(n_steps):
        # Alpha alternates between 0.2 and 0.8 every 10 steps
        # Steps 0-9: alpha=0.2, steps 10-19: alpha=0.8, steps 20-29: alpha=0.2, etc.
        cycle_position = (step // 10) % 2
        alpha = 0.2 if cycle_position == 0 else 0.8
        
        # Randomly select two different vectors from memory
        n_memory = memory.shape[0]
        idx_i = torch.randint(0, n_memory, (1,)).item()
        idx_j = torch.randint(0, n_memory, (1,)).item()
        # Ensure we select different vectors
        while idx_j == idx_i and n_memory > 1:
            idx_j = torch.randint(0, n_memory, (1,)).item()
        x_i = memory[idx_i]
        x_j = memory[idx_j]
        
        # Generate random noise vector
        noise = torch.randn(dim)
        
        # Reorganize: combine vectors with noise scaled by (1-alpha)
        output = reorganize(x_i, x_j, alpha, noise)
        
        # Compute novelty: minimum distance to memory vectors
        novelty = compute_novelty(output, memory)
        
        # Compute coherence: cosine similarity to input vectors
        coherence = compute_coherence(output, x_i, x_j)
        
        # Compute creativity: product of novelty and coherence
        creativity = novelty * coherence
        
        # Store creativity score
        creativity_scores.append(creativity.item())
        
        # Append output to memory for next iterations
        memory = torch.cat([memory, output.unsqueeze(0)], dim=0)
        
        # Print progress every 20 steps
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}/{n_steps}: "
                  f"alpha={alpha:.1f}, "
                  f"novelty={novelty.item():.4f}, "
                  f"coherence={coherence.item():.4f}, "
                  f"creativity={creativity.item():.4f}")
    
    # Plot creativity vs step
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_steps + 1), creativity_scores, linewidth=2)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Creativity (Novelty × Coherence)', fontsize=12)
    plt.title('Creativity Score Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines to show alpha transitions
    for i in range(0, n_steps, 10):
        alpha_val = 0.2 if (i // 10) % 2 == 0 else 0.8
        plt.axvline(x=i, color='red', linestyle='--', alpha=0.2)
    
    # Add legend for alpha values
    plt.text(5, max(creativity_scores) * 0.95, 'α=0.2', fontsize=10, color='blue')
    plt.text(15, max(creativity_scores) * 0.95, 'α=0.8', fontsize=10, color='blue')
    
    plt.tight_layout()
    plt.savefig('creativity_plot.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'creativity_plot.png'")
    print(f"Final memory size: {memory.shape[0]} vectors")
    print(f"Average creativity score: {sum(creativity_scores) / len(creativity_scores):.4f}")
    plt.show()


if __name__ == "__main__":
    main()
