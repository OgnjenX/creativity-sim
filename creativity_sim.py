"""
Creativity Simulation using PyTorch

This script simulates creativity by combining latent vectors with varying degrees of
exploration (controlled by alpha) and measuring the novelty and coherence of the
resulting combinations. It includes:

- Distinct sampling of memory vectors for recombination
- Separate live and replay memories with preferential sampling in undirected mode
- Dynamic/continuous alpha control (cosine schedule or adaptive controller)
- Refined coherence metric (cosine similarity clipped to [0,1])
- Fixed-size memory buffers and tracking of memory growth
- Enhanced plotting of novelty, coherence, competence, creativity (and optional memory size)
"""

import math
from collections import deque
from typing import Tuple, Optional, Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor

LEGEND_LOC_UPPER_RIGHT = "upper right"


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


def _sample_memory_for_normalization(
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


def _avg_pairwise_distance(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the average nonzero pairwise distance within a set of vectors."""
    if len(tensor) <= 1:
        return torch.tensor(0.0, dtype=tensor.dtype)
    dists = torch.cdist(tensor, tensor)
    nonzero = dists[dists > 0]
    return torch.mean(nonzero) if nonzero.numel() > 0 else torch.tensor(0.0, dtype=tensor.dtype)


def _knn_distance(
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
        sample_memory = _sample_memory_for_normalization(
            memory, max_samples=200, deterministic=deterministic_sampling, seed=seed
        )
        avg_distance = _avg_pairwise_distance(sample_memory)
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
        kth, local_mean = _knn_distance(
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
        sample_memory = _sample_memory_for_normalization(
            memory, max_samples=100, deterministic=deterministic_sampling, seed=seed
        )
        avg_distance = _avg_pairwise_distance(sample_memory)
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
    sample_for_div = _sample_memory_for_normalization(
        memory_after, max_samples=200, deterministic=False, seed=None
    )
    diversity_now = float(_avg_pairwise_distance(sample_for_div).item())
    diversity_norm = (
        0.0 if baseline_diversity <= 0 else diversity_now / (baseline_diversity + eps)
    )

    if prev_centroid is None or prev_diversity_norm is None:
        return 1.0, centroid_now, diversity_norm

    delta_div = diversity_norm - float(prev_diversity_norm)
    shift = torch.norm(centroid_now - prev_centroid, dim=0)
    shift_norm = float(shift / (baseline_diversity + eps))
    structure_delta = w_div * delta_div + w_shift * shift_norm
    competence = 1.0 + float(scale * torch.tanh(torch.tensor(structure_delta)))

    return competence, centroid_now, diversity_norm


class AlphaController:
    """Controls alpha as a continuous variable.

    Modes:
        - 'cosine': smooth periodic variation between [alpha_min, alpha_max]
        - 'adaptive': adjusts alpha based on recent change in EMA of creativity

    Args:
        mode: Control mode ('cosine' or 'adaptive').
        alpha_min: Minimum allowed alpha value (exploration-heavy).
        alpha_max: Maximum allowed alpha value (exploitation-heavy).
        period: Period of cosine cycle in steps (used only in 'cosine' mode).
        ema_beta: EMA smoothing factor for creativity tracking (used only in 'adaptive' mode).
        adapt_step_up: Step size to increase alpha when creativity improves.
        adapt_step_down: Step size to decrease alpha when creativity declines.
    """

    def __init__(
        self,
        mode: str = "cosine",
        alpha_min: float = 0.2,
        alpha_max: float = 0.8,
        period: int = 50,
        ema_beta: float = 0.9,
        adapt_step_up: float = 0.02,
        adapt_step_down: float = 0.04,
    ) -> None:
        self.mode = mode
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.period = max(1, period)
        self.ema_beta = ema_beta
        self.adapt_step_up = adapt_step_up
        self.adapt_step_down = adapt_step_down
        self._ema = None
        self._prev_ema = None
        self._alpha = (alpha_min + alpha_max) / 2.0

    @property
    def alpha(self) -> float:
        """Current alpha value."""
        return float(self._alpha)

    def update(self, step: int, creativity: float) -> float:
        """Update alpha based on the current mode and observed creativity.

        Args:
            step: Current simulation step.
            creativity: Latest creativity score.

        Returns:
            Updated alpha value.

        Note:
            In adaptive mode, the first call (typically step=0) initializes the EMA
            with the provided creativity value. Subsequent calls use the EMA
            to adjust alpha based on creativity trends.
        """
        if self.mode == "cosine":
            phase = 2 * math.pi * (step % self.period) / self.period
            self._alpha = self.alpha_min + 0.5 * (1 - math.cos(phase)) * (
                self.alpha_max - self.alpha_min
            )
        elif self.mode == "adaptive":
            # Update EMA
            if self._ema is None:
                self._ema = creativity
                self._prev_ema = creativity
            else:
                self._prev_ema = self._ema
                self._ema = self.ema_beta * self._ema + (1 - self.ema_beta) * creativity
            delta = self._ema - self._prev_ema
            # If creativity is improving, exploit a bit more (increase alpha),
            # else explore (decrease alpha)
            if delta > 0:
                self._alpha = min(self.alpha_max, self._alpha + self.adapt_step_up)
            else:
                self._alpha = max(self.alpha_min, self._alpha - self.adapt_step_down)
        else:
            # Fallback: keep alpha fixed mid-range
            self._alpha = (self.alpha_min + self.alpha_max) / 2.0
        return float(self._alpha)


class MemoryBuffers:
    """Manages live and replay memories with fixed capacities and biased sampling.

    Args:
        dim: Dimensionality of stored vectors.
        live_capacity: Maximum number of vectors in the live buffer.
        replay_capacity: Maximum number of vectors in the replay buffer.
        use_tensor_memory: If True, use tensor-based memory for better
        performance with large buffers.
    """

    def __init__(
        self,
        dim: int,
        live_capacity: int = 256,
        replay_capacity: int = 2048,
        use_tensor_memory: bool = False,
    ):
        self.dim = dim
        self.use_tensor_memory = use_tensor_memory
        # Diversity-preserving config (defaults; may be overridden externally)
        self.memory_policy: str = "simple"  # "simple" | "diverse"
        self.similarity_threshold: float = 0.4  # normalized distance threshold
        self.knn_k: int = 5

        if use_tensor_memory:
            # Tensor-based memory for better performance with large buffers
            self.live_capacity = live_capacity
            self.replay_capacity = replay_capacity
            self._live_tensor = torch.empty(0, dim)
            self._replay_tensor = torch.empty(0, dim)
            self._live_idx = 0  # Rolling index for live buffer
            self._replay_idx = 0  # Rolling index for replay buffer
            self._live_size = 0
            self._replay_size = 0
        else:
            # Original deque-based implementation
            self.live = deque(maxlen=live_capacity)
            self.replay = deque(maxlen=replay_capacity)

    def initialize(self, n_initial: int, replay_fraction: float = 0.3) -> None:
        """Populate the live buffer with random vectors and optionally replay buffer.

        Args:
            n_initial: Number of initial vectors to generate.
            replay_fraction: Fraction of initial vectors to place directly in replay buffer.
        """
        init = torch.randn(n_initial, self.dim)

        if self.use_tensor_memory:
            # Initialize tensor-based memory
            n_replay = int(n_initial * replay_fraction)
            n_live = n_initial - n_replay

            # Initialize tensors with proper capacity
            self._live_tensor = torch.zeros(self.live_capacity, self.dim)
            self._replay_tensor = torch.zeros(self.replay_capacity, self.dim)

            # Add to replay buffer first
            for i in range(n_replay):
                self._replay_tensor[i % self.replay_capacity] = init[i]
            self._replay_size = min(n_replay, self.replay_capacity)
            self._replay_idx = n_replay % self.replay_capacity

            # Add remaining to live buffer
            for i in range(n_replay, n_initial):
                self._live_tensor[i % self.live_capacity] = init[i]
            self._live_size = min(n_live, self.live_capacity)
            self._live_idx = n_live % self.live_capacity
        else:
            # Original deque-based initialization
            n_replay = int(n_initial * replay_fraction)

            # Add to replay buffer first
            for i in range(n_replay):
                self.replay.append(init[i])

            # Add remaining to live buffer
            for i in range(n_replay, n_initial):
                self.live.append(init[i])

    def add(self, vector: torch.Tensor) -> None:
        """Add a vector to the live buffer, moving the oldest to replay if at capacity.

        Args:
            vector: Vector to add.
        """
        if self.memory_policy == "diverse":
            self._add_diverse(vector)
            return

        if self.use_tensor_memory:
            self._add_tensor_memory(vector.detach())
        else:
            self._add_deque_memory(vector.detach())

    def _add_diverse(self, vector: torch.Tensor) -> None:
        """Diversity-preserving insertion with similarity check and replacement.

        - If too similar to existing memory (min distance < threshold), either reject
          or replace the most redundant memory vector (smallest k-NN distance).
        - Maintains fixed capacities and keeps recent samples in live by default.
        """
        v = vector.detach()
        mem = self.full_memory()
        if mem.numel() == 0:
            # No memory yet, fall back to simple add
            return self._add_simple(v)
        # Normalize distances by average pairwise distance for stability
        _, _ = _knn_distance(v, mem, k=self.knn_k, normalize=True)
        min_dist = torch.min(torch.norm(mem - v.unsqueeze(0), dim=1))
        # Similarity threshold applied on normalized scale if possible
        is_too_similar = bool(min_dist.item() < self.similarity_threshold)

        if not is_too_similar:
            return self._add_simple(v)

        # Compute redundancy scores for existing memory: smaller k-NN => denser
        # For moderate sizes this is fine; for very large, could be sampled.
        with torch.no_grad():
            # Full pairwise distances
            dmat = torch.cdist(mem, mem)
            # Mask self distances
            dmat[dmat == 0] = float("inf")
            # k-th nearest neighbor distance for each memory point
            k_eff = min(self.knn_k, dmat.shape[0] - 1) if dmat.shape[0] > 1 else 1
            kth_all, _ = torch.topk(dmat, k=k_eff, largest=False, dim=1)
            kth_scores = kth_all[:, -1]
            # Identify index of most redundant (smallest kth distance)
            replace_idx = int(torch.argmin(kth_scores, dim=0).item())

        # Replace at global memory index
        self._replace_global_index(replace_idx, v)
        return None

    def _add_simple(self, v: torch.Tensor) -> None:
        if self.use_tensor_memory:
            self._add_tensor_memory(v)
        else:
            self._add_deque_memory(v)

    def _add_tensor_memory(self, v: torch.Tensor) -> None:
        if self._live_size == self.live_capacity:
            oldest_idx = self._live_idx
            oldest_vector = self._live_tensor[oldest_idx]
            self._replay_tensor[self._replay_idx] = oldest_vector
            self._replay_idx = (self._replay_idx + 1) % self.replay_capacity
            self._replay_size = min(self._replay_size + 1, self.replay_capacity)
        self._live_tensor[self._live_idx] = v
        self._live_idx = (self._live_idx + 1) % self.live_capacity
        self._live_size = min(self._live_size + 1, self.live_capacity)

    def _add_deque_memory(self, v: torch.Tensor) -> None:
        if len(self.live) == self.live.maxlen:
            oldest = self.live[0]
            self.replay.append(oldest)
        self.live.append(v)

    def _replace_global_index(self, global_idx: int, v: torch.Tensor) -> None:
        """Replace a vector by its index in concatenated [replay, live] memory."""
        if self.use_tensor_memory:
            # Replay comes first in full_memory()
            if global_idx < self._replay_size:
                self._replay_tensor[
                    (self._replay_idx - self._replay_size + global_idx) % self.replay_capacity
                ] = v
            else:
                live_pos = global_idx - self._replay_size
                self._live_tensor[
                    (self._live_idx - self._live_size + live_pos) % self.live_capacity
                ] = v
        else:
            if global_idx < len(self.replay):
                self.replay[global_idx] = v
            else:
                live_pos = global_idx - len(self.replay)
                self.live[live_pos] = v

    def full_memory(self) -> torch.Tensor:
        """Return a tensor containing all vectors from both buffers."""
        if self.use_tensor_memory:
            # Tensor-based memory retrieval
            if self._live_size == 0 and self._replay_size == 0:
                return torch.empty(0, self.dim)

            # Get live memory (handle circular buffer)
            if self._live_size < self.live_capacity:
                live_memory = self._live_tensor[: self._live_size]
            else:
                # Circular buffer: need to reorder
                live_memory = torch.cat(
                    [
                        self._live_tensor[self._live_idx :],
                        self._live_tensor[: self._live_idx],
                    ],
                    dim=0,
                )

            # Get replay memory (handle circular buffer)
            if self._replay_size < self.replay_capacity:
                replay_memory = self._replay_tensor[: self._replay_size]
            else:
                # Circular buffer: need to reorder
                replay_memory = torch.cat(
                    [
                        self._replay_tensor[self._replay_idx :],
                        self._replay_tensor[: self._replay_idx],
                    ],
                    dim=0,
                )

            return torch.cat([replay_memory, live_memory], dim=0)
        else:
            # Original deque-based memory retrieval
            if len(self.live) == 0 and len(self.replay) == 0:
                return torch.empty(0, self.dim)
            all_vecs = list(self.replay) + list(self.live)
            return torch.stack(all_vecs, dim=0)

    def sizes(self) -> Tuple[int, int, int]:
        """Return sizes of live, replay, and total memory."""
        if self.use_tensor_memory:
            return (
                self._live_size,
                self._replay_size,
                self._live_size + self._replay_size,
            )
        else:
            return len(self.live), len(self.replay), len(self.live) + len(self.replay)

    @staticmethod
    def _draw_from_tensor(_tensor: torch.Tensor, size: int) -> int:
        """Return a random index into the given tensor.

        Note: tensor parameter is not used in implementation but kept for interface consistency.
        The method only needs the size to generate a valid index.
        """
        return int(torch.randint(0, size, (1,)).item())

    @staticmethod
    def _draw_distinct_tensor(
        tensor: torch.Tensor,
        size: int,
        avoid_idx: int,
        fallback_tensor: Optional[torch.Tensor] = None,
        fallback_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Draw a vector from tensor with an index different from avoid_idx.

        Args:
            tensor: Primary tensor to draw from.
            size: Actual size of the tensor (it may be less than capacity).
            avoid_idx: Index to avoid when drawing.
            fallback_tensor: Optional secondary tensor to use if primary cannot
            provide distinct vector.
            fallback_size: Size of fallback tensor.

        Returns:
            A vector distinct from the one at avoid_idx.

        Raises:
            ValueError: If insufficient memory to draw distinct vectors.
        """
        if size >= 2:
            # Efficiently sample a distinct index
            valid_indices = [i for i in range(size) if i != avoid_idx]
            j_idx = int(torch.randint(0, len(valid_indices), (1,)).item())
            return tensor[valid_indices[j_idx]]
        if fallback_tensor is not None and fallback_size is not None and fallback_size > 0:
            j_idx = MemoryBuffers._draw_from_tensor(fallback_tensor, fallback_size)
            return fallback_tensor[j_idx]
        else:
            # Explicit error for insufficient memory
            raise ValueError(
                f"Insufficient memory to draw distinct vectors. "
                f"Primary buffer size: {size}, "
                f"Fallback buffer size: {fallback_size if fallback_size else 0}"
            )

    @staticmethod
    def _draw_from(buf: deque) -> int:
        """Return a random index into the given deque."""
        return int(torch.randint(0, len(buf), (1,)).item())

    @staticmethod
    def _draw_distinct(
        buf: deque, avoid_idx: int, fallback_buf: Optional[deque] = None
    ) -> torch.Tensor:
        """Draw a vector from buf with an index different from avoid_idx.
        If buf has size 1 and fallback_buf is provided, draw from fallback_buf.

        Args:
            buf: Primary buffer to draw from.
            avoid_idx: Index to avoid when drawing.
            fallback_buf: Optional secondary buffer to use if buf cannot provide a distinct vector.

        Returns:
            A vector distinct from the one at avoid_idx.

        Raises:
            ValueError: If insufficient memory to draw distinct vectors.
        """
        if len(buf) >= 2:
            # Efficiently sample a distinct index
            valid_indices = [i for i in range(len(buf)) if i != avoid_idx]
            j_idx = int(torch.randint(0, len(valid_indices), (1,)).item())
            return buf[valid_indices[j_idx]]
        if fallback_buf is not None and len(fallback_buf) > 0:
            j_idx = MemoryBuffers._draw_from(fallback_buf)
            return fallback_buf[j_idx]
        else:
            # Explicit error for insufficient memory
            raise ValueError(
                f"Insufficient memory to draw distinct vectors. "
                f"Primary buffer size: {len(buf)}, "
                f"Fallback buffer size: {len(fallback_buf) if fallback_buf else 0}"
            )

    def sample_pair(
        self,
        alpha: float,
        alpha_replay_thresh: float = 0.4,
        far_pair_prob: float = 0.0,
    ) -> tuple[Any, Tensor] | None:
        """Sample two distinct vectors, preferentially from replay when alpha is low.

        Strategy: compute preference p toward replay when alpha <= threshold.
        Draw first sample accordingly; for second, try to draw from the same buffer
        while ensuring distinctness; fallback to the other buffer if needed.

        Args:
            alpha: Current alpha value controlling exploration vs exploitation.
            alpha_replay_thresh: Threshold below which replay sampling is preferred.
            far_pair_prob: Probability of selecting the second parent as the farthest
            vector from the first across the full memory.

        Returns:
            A tuple of two distinct vectors.
        """
        live_size, replay_size, total_mem = self.sizes()
        assert total_mem >= 2, (
            "Not enough memory to sample two distinct vectors "
            f"(current size: {total_mem}, required: 2)"
        )

        have_replay = replay_size > 0 if self.use_tensor_memory else len(self.replay) > 0
        have_live = live_size > 0 if self.use_tensor_memory else len(self.live) > 0

        use_replay_first = self._should_use_replay_first(
            alpha, alpha_replay_thresh, have_live, have_replay
        )

        # Optionally sample a far pair
        far_pair = self._try_sample_far_pair(
            use_replay_first, live_size, replay_size, far_pair_prob
        )
        if far_pair is not None:
            return far_pair

        if self.use_tensor_memory:
            return self._sample_pair_tensor(
                use_replay_first, live_size, replay_size, have_live, have_replay
            )
        return self._sample_pair_deque(use_replay_first, have_live, have_replay)

    @staticmethod
    def _compute_replay_preference(alpha: float, thresh: float, have_replay: bool) -> float:
        if thresh <= 0:
            raise ValueError(
                "alpha_replay_thresh must be greater than zero"
                " for proper replay preference calculation."
            )
        if not have_replay or alpha > thresh:
            return 0.0
        p = (thresh - alpha) / thresh
        return float(min(max(p, 0.0), 1.0))

    def _should_use_replay_first(
        self,
        alpha: float,
        alpha_replay_thresh: float,
        have_live: bool,
        have_replay: bool,
    ) -> bool:
        p_replay = self._compute_replay_preference(alpha, alpha_replay_thresh, have_replay)
        prefer_replay = bool((torch.rand(()) < p_replay).item())
        return have_replay and (not have_live or prefer_replay)

    def _choose_first_parent(
        self, use_replay_first: bool, live_size: int, replay_size: int
    ) -> Tensor:
        if self.use_tensor_memory:
            if use_replay_first:
                i_idx = MemoryBuffers._draw_from_tensor(self._replay_tensor, replay_size)
                return self._replay_tensor[i_idx]
            i_idx = MemoryBuffers._draw_from_tensor(self._live_tensor, live_size)
            return self._live_tensor[i_idx]
        else:
            if use_replay_first:
                i_idx = self._draw_from(self.replay)
                return self.replay[i_idx]
            i_idx = self._draw_from(self.live)
            return self.live[i_idx]

    def _try_sample_far_pair(
        self,
        use_replay_first: bool,
        live_size: int,
        replay_size: int,
        far_pair_prob: float,
    ) -> tuple[Any, Tensor] | None:
        prob = max(0.0, min(1.0, float(far_pair_prob)))
        if float(torch.rand(()).item()) >= prob:
            return None
        x_i = self._choose_first_parent(use_replay_first, live_size, replay_size)
        mem = self.full_memory()
        if mem.shape[0] < 2:
            return None
        dists = torch.norm(mem - x_i.unsqueeze(0), dim=1)
        j_idx = int(torch.argmax(dists).item())
        x_j = mem[j_idx]
        return x_i, x_j

    def _sample_pair_tensor(
        self,
        use_replay_first: bool,
        live_size: int,
        replay_size: int,
        have_live: bool,
        have_replay: bool,
    ) -> tuple[Any, Tensor]:
        if use_replay_first:
            i_idx = MemoryBuffers._draw_from_tensor(self._replay_tensor, replay_size)
            x_i = self._replay_tensor[i_idx]
            x_j = MemoryBuffers._draw_distinct_tensor(
                self._replay_tensor,
                replay_size,
                i_idx,
                fallback_tensor=self._live_tensor if have_live else None,
                fallback_size=live_size if have_live else None,
            )
            return x_i, x_j
        i_idx = MemoryBuffers._draw_from_tensor(self._live_tensor, live_size)
        x_i = self._live_tensor[i_idx]
        x_j = MemoryBuffers._draw_distinct_tensor(
            self._live_tensor,
            live_size,
            i_idx,
            fallback_tensor=self._replay_tensor if have_replay else None,
            fallback_size=replay_size if have_replay else None,
        )
        return x_i, x_j

    def _sample_pair_deque(
        self, use_replay_first: bool, have_live: bool, have_replay: bool
    ) -> tuple[Any, Tensor]:
        if use_replay_first:
            i_idx = self._draw_from(self.replay)
            x_i = self.replay[i_idx]
            x_j = self._draw_distinct(
                self.replay, i_idx, fallback_buf=self.live if have_live else None
            )
            return x_i, x_j
        i_idx = self._draw_from(self.live)
        x_i = self.live[i_idx]
        x_j = self._draw_distinct(
            self.live, i_idx, fallback_buf=self.replay if have_replay else None
        )
        return x_i, x_j


def _update_alpha_for_step(
    alpha_ctrl: AlphaController, step: int, creativity_log: list[float]
) -> float:
    if step > 0:
        return alpha_ctrl.update(step - 1, creativity_log[-1])
    return alpha_ctrl.alpha


def _select_pair_or_fallback(
    buffers: MemoryBuffers,
    curr_alpha: float,
    alpha_replay_thresh: float,
    dim: int,
    far_pair_prob: float,
) -> tuple[Tensor, Tensor]:
    pair = buffers.sample_pair(
        curr_alpha, alpha_replay_thresh=alpha_replay_thresh, far_pair_prob=far_pair_prob
    )
    if pair is None:
        return torch.randn(dim), torch.randn(dim)
    return pair  # type: ignore[return-value]


def _log_progress(
    step: int,
    n_steps: int,
    curr_alpha: float,
    novelty: float,
    coherence: float,
    competence: float,
    creativity: float,
    live_sz: int,
    replay_sz: int,
    total_sz: int,
) -> None:
    print(
        f"Step {step}/{n_steps}: "
        f"alpha={curr_alpha:.3f}, "
        f"novelty={novelty:.4f}, "
        f"coherence={coherence:.4f}, "
        f"competence={competence:.4f}, "
        f"creativity={creativity:.4f}, "
        f"mem_live={live_sz}, mem_replay={replay_sz}, total_mem={total_sz}"
    )


def _plot_metrics(
    steps,
    creativity_log,
    novelty_log,
    diversity_log,
    competence_log,
    coherence_log,
    mem_live_log,
    mem_replay_log,
    mem_total_log,
    alpha_log,
    alpha_replay_thresh,
    pulse_steps: list[int] | None = None,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(
        steps,
        creativity_log,
        label="Creativity",
        color="tab:purple",
        linewidth=2,
        linestyle="-",
        marker="o",
        markersize=3,
        markevery=max(1, len(steps) // 20),
    )
    axes[0, 0].set_title("Creativity over Steps", fontweight="bold")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Creativity")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc=LEGEND_LOC_UPPER_RIGHT)

    axes[0, 1].plot(
        steps,
        novelty_log,
        label="Novelty",
        color="tab:orange",
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=3,
        markevery=max(1, len(steps) // 20),
    )
    axes[0, 1].set_title("Novelty over Steps", fontweight="bold")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Novelty (k-NN or min)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc=LEGEND_LOC_UPPER_RIGHT)

    # Competence curve (reorganizational)
    axes[0, 2].plot(
        steps,
        competence_log,
        label="Competence",
        color="tab:gray",
        linewidth=2,
        linestyle="-",
        marker="*",
        markersize=3,
        markevery=max(1, len(steps) // 20),
    )
    axes[0, 2].set_title("Competence (structure improvement)", fontweight="bold")
    axes[0, 2].set_xlabel("Step")
    axes[0, 2].set_ylabel("Competence (≈1 baseline)")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend(loc=LEGEND_LOC_UPPER_RIGHT)

    axes[1, 0].plot(
        steps,
        coherence_log,
        label="Coherence",
        color="tab:green",
        linewidth=2,
        linestyle="-.",
        marker="^",
        markersize=3,
        markevery=max(1, len(steps) // 20),
    )
    axes[1, 0].set_title("Coherence over Steps", fontweight="bold")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Coherence (clipped cosine similarity)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc=LEGEND_LOC_UPPER_RIGHT)

    ax = axes[1, 1]
    ax.plot(
        steps,
        mem_total_log,
        label="Total Memory",
        color="tab:blue",
        linewidth=2,
        linestyle="-",
        marker="D",
        markersize=2,
        markevery=max(1, len(steps) // 15),
    )
    ax.plot(
        steps,
        mem_live_log,
        label="Live Memory",
        color="tab:cyan",
        linewidth=1.5,
        linestyle=":",
        marker="o",
        markersize=2,
        markevery=max(1, len(steps) // 15),
        alpha=0.8,
    )
    ax.plot(
        steps,
        mem_replay_log,
        label="Replay Memory",
        color="tab:red",
        linewidth=1.5,
        linestyle="-.",
        marker="s",
        markersize=2,
        markevery=max(1, len(steps) // 15),
        alpha=0.8,
    )
    ax.set_title("Memory Growth and Alpha Control", fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Memory Size")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(
        steps,
        alpha_log,
        label="Alpha",
        color="black",
        linestyle="--",
        linewidth=2,
        marker="x",
        markersize=2,
        markevery=max(1, len(steps) // 10),
        alpha=0.7,
    )
    ax2.axhline(
        y=alpha_replay_thresh,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label=f"Replay Threshold ({alpha_replay_thresh})",
    )
    ax2.set_ylabel("Alpha (Exploration ← → Exploitation)")
    ax2.set_ylim(0, 1)

    memory_lines, memory_labels = ax.get_legend_handles_labels()
    ax.legend(memory_lines, memory_labels, loc="upper left", bbox_to_anchor=(0.02, 0.98))

    alpha_lines, alpha_labels = ax2.get_legend_handles_labels()
    ax2.legend(
        alpha_lines,
        alpha_labels,
        loc=LEGEND_LOC_UPPER_RIGHT,
        bbox_to_anchor=(0.98, 0.98),
    )

    # Memory Diversity panel (post-update)
    axes[1, 2].plot(
        steps,
        diversity_log,
        label="Diversity",
        color="tab:brown",
        linewidth=2,
        linestyle="-",
        marker=".",
        markersize=3,
        markevery=max(1, len(steps) // 20),
    )
    axes[1, 2].set_title("Memory Diversity (avg distance)", fontweight="bold")
    axes[1, 2].set_xlabel("Step")
    axes[1, 2].set_ylabel("Avg pairwise distance")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(loc=LEGEND_LOC_UPPER_RIGHT)

    # Mark pulse steps, if any
    if pulse_steps:
        for s in pulse_steps:
            for row in range(2):
                for col in range(3):
                    axes[row, col].axvline(x=s, color="tab:red", alpha=0.2, linestyle=":")
    fig.suptitle("Creativity Simulation Metrics", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    plt.savefig("creativity_plot.png", dpi=300, bbox_inches="tight")
    print("\nPlot saved as 'creativity_plot.png'")


def _validate_dimensions(dim):
    if dim < 2:
        raise ValueError("Vector dimension must be at least 2 for meaningful creativity simulation")
    if dim > 512:
        print(
            f"⚠️  High dimension ({dim}) detected. "
            "Consider using tensor-based memory for better performance."
        )


def _warn_large_simulation(n_initial_memory, n_steps, use_tensor_memory, dim):
    if (n_initial_memory > 1000 or n_steps > 1000) and not use_tensor_memory:
        print("⚠️  Performance Warning: Large simulation detected.")
        print("   Consider setting use_tensor_memory=True for better performance.")
        print("   This will use tensor-based memory management instead of deque operations.")
    if dim > 512 and not use_tensor_memory:
        print("   Recommendation: Set use_tensor_memory=True for dimensions > 512")


def _setup_simulation_config():
    dim = 16  # Dimension of latent vectors (try 32, 64, 128 for richer dynamics)
    n_initial_memory = 100  # Initial number of memory vectors
    n_steps = 300  # Number of simulation steps

    # Memory buffer capacities
    live_capacity = 256
    replay_capacity = 2048

    # Performance optimization for large simulations
    use_tensor_memory = False  # Set to True for better performance with large buffers

    # Alpha control settings
    alpha_mode = "adaptive"  # "cosine" or "adaptive"
    alpha_min, alpha_max = 0.1, 0.9
    cosine_period = 100

    # Replay sampling preference threshold for undirected mode
    alpha_replay_thresh = 0.4

    # Noise scaling method
    noise_scale = "sqrt"  # "linear", "sqrt", or "adaptive"

    # Novelty computation and diversity options
    novelty_mode = "knn"  # "knn" or "min"
    knn_k = 5
    diversity_lambda = 0.2
    deterministic_novelty_sampling = False  # For reproducible results in high-precision studies
    novelty_sampling_seed = 42  # Seed for deterministic novelty sampling

    # Diversity-preserving memory policy
    memory_policy = "diverse"  # "diverse" or "simple"
    similarity_threshold = 0.4  # normalized distance threshold

    # Parent sampling strategy
    far_pair_prob = 0.2

    # Stagnation-triggered exploration pulse
    creativity_pulse = True
    pulse_window = 30
    pulse_drop_tol = 0.03  # trigger if drop vs. prior window exceeds this
    pulse_steps = 6
    pulse_noise_gain = 1.0  # extra noise magnitude added during pulse
    pulse_alpha_drop = 0.15  # temporarily decrease alpha by this amount

    # Validation
    _validate_dimensions(dim)
    _warn_large_simulation(n_initial_memory, n_steps, use_tensor_memory, dim)

    return {
        "dim": dim,
        "n_initial_memory": n_initial_memory,
        "n_steps": n_steps,
        "live_capacity": live_capacity,
        "replay_capacity": replay_capacity,
        "use_tensor_memory": use_tensor_memory,
        "alpha_mode": alpha_mode,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "cosine_period": cosine_period,
        "alpha_replay_thresh": alpha_replay_thresh,
        "noise_scale": noise_scale,
        "novelty_mode": novelty_mode,
        "knn_k": knn_k,
        "diversity_lambda": diversity_lambda,
        "deterministic_novelty_sampling": deterministic_novelty_sampling,
        "novelty_sampling_seed": novelty_sampling_seed,
        "memory_policy": memory_policy,
        "similarity_threshold": similarity_threshold,
        "far_pair_prob": far_pair_prob,
        "creativity_pulse": creativity_pulse,
        "pulse_window": pulse_window,
        "pulse_drop_tol": pulse_drop_tol,
        "pulse_steps": pulse_steps,
        "pulse_noise_gain": pulse_noise_gain,
        "pulse_alpha_drop": pulse_alpha_drop,
    }


def _initialize_components(config):
    buffers = MemoryBuffers(
        dim=config["dim"],
        live_capacity=config["live_capacity"],
        replay_capacity=config["replay_capacity"],
        use_tensor_memory=config["use_tensor_memory"],
    )
    buffers.initialize(config["n_initial_memory"], replay_fraction=0.3)
    # Apply memory diversity settings
    buffers.memory_policy = config["memory_policy"]
    buffers.similarity_threshold = config["similarity_threshold"]
    buffers.knn_k = config["knn_k"]

    alpha_ctrl = AlphaController(
        mode=config["alpha_mode"],
        alpha_min=config["alpha_min"],
        alpha_max=config["alpha_max"],
        period=config["cosine_period"],
    )

    # Baseline diversity for competence normalization
    init_memory = buffers.full_memory()
    sample_for_div = _sample_memory_for_normalization(
        init_memory, max_samples=200, deterministic=False, seed=None
    )
    config["baseline_diversity"] = float(_avg_pairwise_distance(sample_for_div).item())

    return buffers, alpha_ctrl


def _initialize_logs():
    logs = {
        "novelty_log": [],
        "diversity_log": [],
        "competence_log": [],
        "coherence_log": [],
        "creativity_log": [],
        "alpha_log": [],
        "mem_live_log": [],
        "mem_replay_log": [],
        "mem_total_log": [],
        "pulse_markers": [],
        "pulse_counter": 0,
        # Internal state for competence
        "prev_centroid": None,
        "prev_diversity_norm": None,
    }
    return logs


def _apply_pulse_modulation(
    curr_alpha, pulse_counter, alpha_min, pulse_alpha_drop, pulse_noise_gain
):
    alpha_effective = curr_alpha
    extra_noise = 0.0
    if pulse_counter > 0:
        alpha_effective = max(alpha_min, curr_alpha - pulse_alpha_drop)
        extra_noise = pulse_noise_gain
        pulse_counter -= 1
    return alpha_effective, extra_noise, pulse_counter


def _compute_and_log_metrics(output, buffers, config, x_i, x_j, alpha_effective, logs):
    # Pre-update memory for novelty/coherence
    memory_before = buffers.full_memory()
    novelty = compute_novelty(
        output,
        memory_before,
        mode=config["novelty_mode"],
        k=config["knn_k"],
        diversity_lambda=config["diversity_lambda"],
        deterministic_sampling=config["deterministic_novelty_sampling"],
        seed=config["novelty_sampling_seed"],
    )
    coherence = compute_coherence(output, x_i, x_j)

    # Update memory with new output
    buffers.add(output)

    # Post-update memory for competence and diversity
    memory_after = buffers.full_memory()
    competence, new_centroid, diversity_norm = compute_competence(
        logs.get("prev_centroid"),
        logs.get("prev_diversity_norm"),
        memory_after,
        config.get("baseline_diversity", 0.0),
    )
    creativity = novelty * coherence * competence

    sample_for_div = _sample_memory_for_normalization(
        memory_after, max_samples=200, deterministic=False, seed=None
    )
    diversity = float(_avg_pairwise_distance(sample_for_div).item())

    # Log metrics
    logs["novelty_log"].append(float(novelty.item()))
    logs["competence_log"].append(float(competence))
    logs["diversity_log"].append(diversity)
    logs["coherence_log"].append(float(coherence.item()))
    logs["creativity_log"].append(float(creativity.item()))
    logs["alpha_log"].append(alpha_effective)

    # Update competence state
    logs["prev_centroid"] = new_centroid
    logs["prev_diversity_norm"] = diversity_norm

    # Memory sizes
    live_sz, replay_sz, total_sz = buffers.sizes()
    logs["mem_live_log"].append(live_sz)
    logs["mem_replay_log"].append(replay_sz)
    logs["mem_total_log"].append(total_sz)


def _check_and_trigger_pulse(
    creativity_pulse, step, pulse_window, creativity_log, pulse_drop_tol, pulse_steps, pulse_markers
):
    pulse_counter = 0
    if creativity_pulse and step + 1 >= 2 * pulse_window:
        recent = creativity_log[-pulse_window:]
        prev = creativity_log[-2 * pulse_window : -pulse_window]
        if sum(recent) / pulse_window < (1 - pulse_drop_tol) * (sum(prev) / pulse_window):
            pulse_counter = pulse_steps
            pulse_markers.append(step + 1)
    return pulse_counter


def main():
    """Main simulation loop for creativity exploration.

    Configuration Tips:
    - dim: Higher dimensions (32, 64, 128) provide richer dynamics but may need more memory
    - n_steps: Longer simulations reveal more stable patterns in creativity metrics
    - use_tensor_memory: Recommended for dim > 512 or n_steps > 1000
    - noise_scale: 'sqrt' is default, 'adaptive' for balanced exploration
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)

    config = _setup_simulation_config()
    buffers, alpha_ctrl = _initialize_components(config)
    logs = _initialize_logs()

    # Main simulation loop
    for step in range(config["n_steps"]):
        curr_alpha = _update_alpha_for_step(alpha_ctrl, step, logs["creativity_log"])
        alpha_effective, extra_noise, logs["pulse_counter"] = _apply_pulse_modulation(
            curr_alpha,
            logs["pulse_counter"],
            config["alpha_min"],
            config["pulse_alpha_drop"],
            config["pulse_noise_gain"],
        )
        x_i, x_j = _select_pair_or_fallback(
            buffers,
            alpha_effective,
            config["alpha_replay_thresh"],
            config["dim"],
            config["far_pair_prob"],
        )
        noise = torch.randn(config["dim"])
        if extra_noise > 0:
            noise = noise + extra_noise * torch.randn(config["dim"])
        output = reorganize(x_i, x_j, alpha_effective, noise, noise_scale=config["noise_scale"])
        _compute_and_log_metrics(output, buffers, config, x_i, x_j, alpha_effective, logs)
        logs["pulse_counter"] = _check_and_trigger_pulse(
            config["creativity_pulse"],
            step,
            config["pulse_window"],
            logs["creativity_log"],
            config["pulse_drop_tol"],
            config["pulse_steps"],
            logs["pulse_markers"],
        )
        if (step + 1) % 20 == 0:
            _log_progress(
                step + 1,
                config["n_steps"],
                alpha_effective,
                logs["novelty_log"][-1],
                logs["coherence_log"][-1],
                logs["competence_log"][-1] if logs["competence_log"] else 1.0,
                logs["creativity_log"][-1],
                logs["mem_live_log"][-1],
                logs["mem_replay_log"][-1],
                logs["mem_total_log"][-1],
            )

    # Plotting
    steps = list(range(1, config["n_steps"] + 1))
    _plot_metrics(
        steps,
        logs["creativity_log"],
        logs["novelty_log"],
        logs["diversity_log"],
        logs["competence_log"],
        logs["coherence_log"],
        logs["mem_live_log"],
        logs["mem_replay_log"],
        logs["mem_total_log"],
        logs["alpha_log"],
        config["alpha_replay_thresh"],
        logs["pulse_markers"],
    )
    print(
        f"Final memory sizes: live={logs['mem_live_log'][-1]}, "
        f"replay={logs['mem_replay_log'][-1]}, total={logs['mem_total_log'][-1]}"
    )
    print(
        f"Average creativity score: {sum(logs['creativity_log']) / len(logs['creativity_log']):.4f}"
    )
    plt.show()


if __name__ == "__main__":
    main()
