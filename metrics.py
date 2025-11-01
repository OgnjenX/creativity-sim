"""Creativity metrics and utility functions.

This module contains core metric computations for novelty, coherence, and competence,
along with helper functions for vector operations and memory utilities.
"""

import math
from typing import Tuple, Optional

import torch
from torch import Tensor


def reorganize(
    x_i: torch.Tensor,
    x_j: torch.Tensor,
    alpha: float,
    noise: torch.Tensor,
    noise_scale: str = "sqrt",
) -> torch.Tensor:
    """Combine two input vectors with weighted averaging and scaled noise.

    Args:
        x_i: First input vector.
        x_j: Second input vector.
        alpha: Weighting factor for combination in [0, 1]; higher values favor x_i.
        noise: Random noise vector for exploration.
        noise_scale: Method for scaling noise ('linear', 'sqrt', 'adaptive').

    Returns:
        Combined vector with noise scaled according to the chosen method.
    """
    combined = alpha * x_i + (1 - alpha) * x_j

    if noise_scale == "linear":
        # Original linear scaling: (1 - alpha)
        noise_factor = 1 - alpha
    elif noise_scale == "sqrt":
        # Square root scaling: sqrt(1 - alpha) - less aggressive at low alpha
        noise_factor = math.sqrt(1 - alpha)
    elif noise_scale == "adaptive":
        # Adaptive scaling that prevents noise domination
        # Noise is maximum at alpha=0.5, decreases toward both extremes
        noise_factor = 0.5 * (1 + math.cos(2 * math.pi * (alpha - 0.5)))
    else:
        raise ValueError(f"Unknown noise_scale: {noise_scale}")

    result = combined + noise_factor * noise
    return result


def sample_memory_for_normalization(
    memory: torch.Tensor,
    max_samples: int,
    deterministic: bool,
    seed: Optional[int],
) -> torch.Tensor:
    """Return a possibly downsampled memory tensor for normalization.

    Downsamples to at most ``max_samples`` vectors, using either deterministic
    evenly spaced sampling or random permutation.
    """
    if len(memory) <= max_samples:
        return memory
    if deterministic:
        if seed is not None:
            torch.manual_seed(seed)
        step = len(memory) // max_samples
        indices = torch.arange(0, len(memory), step)[:max_samples]
        return memory[indices]
    indices = torch.randperm(len(memory))[:max_samples]
    return memory[indices]


def avg_pairwise_distance(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the average nonzero pairwise distance within a set of vectors."""
    if len(tensor) <= 1:
        return torch.tensor(0.0, dtype=tensor.dtype)
    dists = torch.cdist(tensor, tensor)
    nonzero = dists[dists > 0]
    return torch.mean(nonzero) if nonzero.numel() > 0 else torch.tensor(0.0, dtype=tensor.dtype)


def knn_distance(
    vector: torch.Tensor,
    memory: torch.Tensor,
    k: int = 5,
    normalize: bool = True,
    deterministic_sampling: bool = False,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return k-NN novelty and a simple local-density proxy.

    - Novelty: distance to the k-th nearest neighbor (or mean of top-k if <k).
    - Local density: mean distance of k nearest neighbors (smaller => denser).
    Optionally normalize distances by average pairwise distance of a sampled memory.
    """
    if memory.numel() == 0:
        return torch.tensor(float("inf"), dtype=vector.dtype), torch.tensor(0.0)
    # Distances to all memory items
    dists = torch.norm(memory - vector.unsqueeze(0), dim=1)
    # Sort distances ascending
    d_sorted, _ = torch.sort(dists)
    k_eff = min(k, len(d_sorted))
    # k-NN novelty proxy
    kth = d_sorted[k_eff - 1]
    # Local density proxy: average of k nearest distances
    local_mean = torch.mean(d_sorted[:k_eff])

    if normalize and len(memory) > 1:
        sample_memory = sample_memory_for_normalization(
            memory, max_samples=200, deterministic=deterministic_sampling, seed=seed
        )
        avg_distance = avg_pairwise_distance(sample_memory)
        if avg_distance > 0:
            kth = kth / avg_distance
            local_mean = local_mean / avg_distance
    return kth, local_mean


def compute_novelty(
    vector: torch.Tensor,
    memory: torch.Tensor,
    normalize: bool = True,
    mode: str = "min",
    k: int = 5,
    diversity_lambda: float = 0.2,
    deterministic_sampling: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Compute novelty metric.

    Modes:
    - "min": minimum distance (legacy)
    - "knn": k-NN novelty with optional diversity bonus

    Args:
        vector: Query vector to evaluate.
        memory: Collection of memory vectors.
        normalize: Whether to normalize distances by average memory distance.
        mode: "min" or "knn".
        k: Number of neighbors for k-NN novelty.
        diversity_lambda: Weight of diversity bonus; set 0 to disable.
        deterministic_sampling: If True, use deterministic sampling for reproducible results.
        seed: Random seed for deterministic sampling (only used if deterministic_sampling=True).

    Returns:
        Novelty score (higher is more novel); infinity if memory is empty.
    """
    if memory.numel() == 0:
        return torch.tensor(float("inf"), dtype=vector.dtype)

    if mode == "knn":
        kth, local_mean = knn_distance(
            vector,
            memory,
            k=k,
            normalize=normalize,
            deterministic_sampling=deterministic_sampling,
            seed=seed,
        )
        # Convert local density to [0,1]-ish sparsity proxy: higher when sparser
        # Avoid division by zero by adding small epsilon
        eps = 1e-8
        sparsity = kth / (local_mean + eps)
        novelty = kth + diversity_lambda * sparsity
        return novelty

    # Legacy minimum-distance novelty
    distances = torch.norm(memory - vector.unsqueeze(0), dim=1)
    min_distance = torch.min(distances)
    if normalize and len(memory) > 1:
        sample_memory = sample_memory_for_normalization(
            memory, max_samples=100, deterministic=deterministic_sampling, seed=seed
        )
        avg_distance = avg_pairwise_distance(sample_memory)
        if avg_distance > 0:
            return min_distance / avg_distance
    return min_distance


def compute_coherence(vector: torch.Tensor, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
    """Compute coherence as the average cosine similarity to input vectors, clipped to [0, 1].

    Args:
        vector: Output vector to evaluate.
        x_i: First input vector.
        x_j: Second input vector.

    Returns:
        Average cosine similarity to inputs, clipped to [0, 1].
    """
    # Use top-level torch.cosine_similarity on 1D vectors for clarity and
    # to avoid static analysis false positives on torch.nn.functional.
    cos_sim_i = torch.cosine_similarity(vector, x_i, dim=0)
    cos_sim_j = torch.cosine_similarity(vector, x_j, dim=0)
    coherence = (cos_sim_i + cos_sim_j) / 2.0
    # Clip to [0, 1]
    coherence = torch.clamp(coherence, 0.0, 1.0)
    return coherence


def compute_competence(
    prev_centroid: Optional[Tensor],
    prev_diversity_norm: Optional[float],
    memory_after: Tensor,
    baseline_diversity: float,
    *,
    w_div: float = 0.7,
    w_shift: float = 0.3,
    scale: float = 0.5,
) -> Tuple[float, Tensor, float]:
    """Compute competence as improvement in representational structure.

    Combines two post-update signals:
    - Diversity change: increase in average pairwise distance (normalized).
    - Prototype (centroid) shift: magnitude of centroid change.

    competence = 1 + scale * tanh(w_div*delta_div + w_shift*shift_norm)

    Returns (competence, new_centroid, diversity_norm).
    """
    eps = 1e-8
    if memory_after.numel() == 0:
        return 1.0, torch.zeros(0), 0.0

    centroid_now = torch.mean(memory_after, dim=0)
    sample_for_div = sample_memory_for_normalization(
        memory_after, max_samples=200, deterministic=False, seed=None
    )
    diversity_now = float(avg_pairwise_distance(sample_for_div).item())
    diversity_norm = 0.0 if baseline_diversity <= 0 else diversity_now / (baseline_diversity + eps)

    if prev_centroid is None or prev_diversity_norm is None:
        return 1.0, centroid_now, diversity_norm

    delta_div = diversity_norm - float(prev_diversity_norm)
    shift = torch.norm(centroid_now - prev_centroid, dim=0)
    shift_norm = float(shift / (baseline_diversity + eps))
    structure_delta = w_div * delta_div + w_shift * shift_norm
    competence = 1.0 + float(scale * torch.tanh(torch.tensor(structure_delta)))

    return competence, centroid_now, diversity_norm
