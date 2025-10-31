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
- Enhanced plotting of novelty, coherence, creativity (and optional memory size)
"""

import math
from collections import deque
from typing import Tuple, Optional, Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor


def reorganize(x_i: torch.Tensor, x_j: torch.Tensor, alpha: float, noise: torch.Tensor,
               noise_scale: str = "sqrt") -> torch.Tensor:
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
        noise_factor = (1 - alpha)
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


def compute_novelty(vector: torch.Tensor, memory: torch.Tensor, normalize: bool = True,
                    deterministic_sampling: bool = False, seed: Optional[int] = None) -> torch.Tensor:
    """Compute novelty as the minimum Euclidean distance to all vectors in memory.

    Optionally normalizes by average memory distance to prevent systematic bias
    as memory grows.

    Args:
        vector: Query vector to evaluate.
        memory: Collection of memory vectors.
        normalize: Whether to normalize novelty by average memory distance.
        deterministic_sampling: If True, use deterministic sampling for reproducible results.
        seed: Random seed for deterministic sampling (only used if deterministic_sampling=True).

    Returns:
        Normalized minimum Euclidean distance to memory; infinity if memory is empty.
    """
    if memory.numel() == 0:
        # Return infinity to indicate maximal novelty when memory is empty
        return torch.tensor(float('inf'), dtype=vector.dtype)
    distances = torch.norm(memory - vector.unsqueeze(0), dim=1)
    min_distance = torch.min(distances)

    if normalize and len(memory) > 1:
        # Normalize by average pairwise distance in memory to prevent compression bias
        # Sample a subset for efficiency if memory is large
        sample_size = min(len(memory), 100)
        if len(memory) > sample_size:
            if deterministic_sampling:
                if seed is not None:
                    torch.manual_seed(seed)
                # Use deterministic sampling: take evenly spaced indices
                step = len(memory) // sample_size
                indices = torch.arange(0, len(memory), step)[:sample_size]
            else:
                indices = torch.randperm(len(memory))[:sample_size]
            sample_memory = memory[indices]
        else:
            sample_memory = memory

        # Compute average pairwise distance in sample
        if len(sample_memory) > 1:
            pairwise_distances = torch.cdist(sample_memory, sample_memory)
            avg_distance = torch.mean(pairwise_distances[pairwise_distances > 0])
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
    cos_sim_i = torch.nn.functional.cosine_similarity(vector.unsqueeze(0), x_i.unsqueeze(0))[0]
    cos_sim_j = torch.nn.functional.cosine_similarity(vector.unsqueeze(0), x_j.unsqueeze(0))[0]
    coherence = (cos_sim_i + cos_sim_j) / 2.0
    # Clip to [0, 1]
    coherence = torch.clamp(coherence, 0.0, 1.0)
    return coherence


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
            self._alpha = self.alpha_min + 0.5 * (1 - math.cos(phase)) * (self.alpha_max - self.alpha_min)
        elif self.mode == "adaptive":
            # Update EMA
            if self._ema is None:
                self._ema = creativity
                self._prev_ema = creativity
            else:
                self._prev_ema = self._ema
                self._ema = self.ema_beta * self._ema + (1 - self.ema_beta) * creativity
            delta = self._ema - self._prev_ema
            # If creativity is improving, exploit a bit more (increase alpha), else explore (decrease alpha)
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
        use_tensor_memory: If True, use tensor-based memory for better performance with large buffers.
    """

    def __init__(self, dim: int, live_capacity: int = 256, replay_capacity: int = 2048,
                 use_tensor_memory: bool = False):
        self.dim = dim
        self.use_tensor_memory = use_tensor_memory

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
        if self.use_tensor_memory:
            # Tensor-based memory management
            if self._live_size == self.live_capacity:
                # Move oldest from live to replay
                oldest_idx = self._live_idx
                oldest_vector = self._live_tensor[oldest_idx]

                # Add to replay buffer
                self._replay_tensor[self._replay_idx] = oldest_vector
                self._replay_idx = (self._replay_idx + 1) % self.replay_capacity
                self._replay_size = min(self._replay_size + 1, self.replay_capacity)

            # Add new vector to live buffer
            self._live_tensor[self._live_idx] = vector.detach()
            self._live_idx = (self._live_idx + 1) % self.live_capacity
            self._live_size = min(self._live_size + 1, self.live_capacity)
        else:
            # Original deque-based memory management
            if len(self.live) == self.live.maxlen:
                oldest = self.live[0]
                self.replay.append(oldest)
            self.live.append(vector.detach())

    def full_memory(self) -> torch.Tensor:
        """Return a tensor containing all vectors from both buffers."""
        if self.use_tensor_memory:
            # Tensor-based memory retrieval
            if self._live_size == 0 and self._replay_size == 0:
                return torch.empty(0, self.dim)

            # Get live memory (handle circular buffer)
            if self._live_size < self.live_capacity:
                live_memory = self._live_tensor[:self._live_size]
            else:
                # Circular buffer: need to reorder
                live_memory = torch.cat([
                    self._live_tensor[self._live_idx:],
                    self._live_tensor[:self._live_idx]
                ], dim=0)

            # Get replay memory (handle circular buffer)
            if self._replay_size < self.replay_capacity:
                replay_memory = self._replay_tensor[:self._replay_size]
            else:
                # Circular buffer: need to reorder
                replay_memory = torch.cat([
                    self._replay_tensor[self._replay_idx:],
                    self._replay_tensor[:self._replay_idx]
                ], dim=0)

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
            return self._live_size, self._replay_size, self._live_size + self._replay_size
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
    def _draw_distinct_tensor(tensor: torch.Tensor, size: int, avoid_idx: int,
                             fallback_tensor: Optional[torch.Tensor] = None,
                             fallback_size: Optional[int] = None) -> torch.Tensor:
        """Draw a vector from tensor with an index different from avoid_idx.

        Args:
            tensor: Primary tensor to draw from.
            size: Actual size of the tensor (it may be less than capacity).
            avoid_idx: Index to avoid when drawing.
            fallback_tensor: Optional secondary tensor to use if primary cannot provide distinct vector.
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
            raise ValueError(f"Insufficient memory to draw distinct vectors. "
                           f"Primary buffer size: {size}, "
                           f"Fallback buffer size: {fallback_size if fallback_size else 0}")

    @staticmethod
    def _draw_from(buf: deque) -> int:
        """Return a random index into the given deque."""
        return int(torch.randint(0, len(buf), (1,)).item())

    @staticmethod
    def _draw_distinct(buf: deque, avoid_idx: int, fallback_buf: Optional[deque] = None) -> torch.Tensor:
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
            raise ValueError(f"Insufficient memory to draw distinct vectors. "
                           f"Primary buffer size: {len(buf)}, "
                           f"Fallback buffer size: {len(fallback_buf) if fallback_buf else 0}")

    def sample_pair(self, alpha: float, alpha_replay_thresh: float = 0.4) -> tuple[Any, Tensor] | None:
        """Sample two distinct vectors, preferentially from replay when alpha is low.

        Strategy: compute preference p toward replay when alpha <= threshold.
        Draw first sample accordingly; for second, try to draw from the same buffer
        while ensuring distinctness; fallback to the other buffer if needed.

        Args:
            alpha: Current alpha value controlling exploration vs exploitation.
            alpha_replay_thresh: Threshold below which replay sampling is preferred.

        Returns:
            A tuple of two distinct vectors.
        """
        # Common logic for both tensor and deque implementations
        live_size, replay_size, total_mem = self.sizes()
        assert total_mem >= 2, f"Not enough memory to sample two distinct vectors (current size: {total_mem}, required: 2)"
        have_replay = replay_size > 0 if self.use_tensor_memory else len(self.replay) > 0
        have_live = live_size > 0 if self.use_tensor_memory else len(self.live) > 0

        # Preference for replay when alpha is small
        p_replay = 0.0
        if alpha_replay_thresh <= 0:
            raise ValueError("alpha_replay_thresh must be greater than zero for proper replay preference calculation.")
        if alpha <= alpha_replay_thresh and have_replay:
            p_replay = (alpha_replay_thresh - alpha) / alpha_replay_thresh
            p_replay = float(min(max(p_replay, 0.0), 1.0))

        # Decide source for first draw
        use_replay_first = have_replay and (not have_live or torch.rand(()) < p_replay)

        if self.use_tensor_memory:
            # Tensor-based sampling
            if use_replay_first:
                i_idx = MemoryBuffers._draw_from_tensor(self._replay_tensor, replay_size)
                x_i = self._replay_tensor[i_idx]
                # Second draw: prefer replay if possible but ensure distinct index
                x_j = MemoryBuffers._draw_distinct_tensor(
                    self._replay_tensor, replay_size, i_idx,
                    fallback_tensor=self._live_tensor if have_live else None,
                    fallback_size=live_size if have_live else None
                )
                return x_i, x_j
            else:
                i_idx = MemoryBuffers._draw_from_tensor(self._live_tensor, live_size)
                x_i = self._live_tensor[i_idx]
                # Second draw: try live first
                x_j = MemoryBuffers._draw_distinct_tensor(
                    self._live_tensor, live_size, i_idx,
                    fallback_tensor=self._replay_tensor if have_replay else None,
                    fallback_size=replay_size if have_replay else None
                )
                return x_i, x_j
        else:
            # Original deque-based sampling
            if use_replay_first:
                i_idx = self._draw_from(self.replay)
                x_i = self.replay[i_idx]
                # Second draw: prefer replay if possible but ensure distinct index
                x_j = self._draw_distinct(self.replay, i_idx, fallback_buf=self.live if have_live else None)
            else:
                i_idx = self._draw_from(self.live)
                x_i = self.live[i_idx]
                # Second draw: try live first
                x_j = self._draw_distinct(self.live, i_idx, fallback_buf=self.replay if have_replay else None)

            return x_i, x_j


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

    # === EXPERIMENT CONFIGURATION ===
    # Try different configurations to explore creativity dynamics:
    # - Small: dim=8, n_steps=200 (quick tests)
    # - Medium: dim=16, n_steps=300 (default)
    # - Large: dim=64, n_steps=1000, use_tensor_memory=True (rich dynamics)
    # - XL: dim=128, n_steps=2000, use_tensor_memory=True (complex behavior)

    dim = 16  # Dimension of latent vectors (try 32, 64, 128 for richer dynamics)
    n_initial_memory = 100  # Initial number of memory vectors
    n_steps = 300  # Number of simulation steps

    # Dimension testing and validation
    if dim < 2:
        raise ValueError("Vector dimension must be at least 2 for meaningful creativity simulation")
    if dim > 512:
        print(f"⚠️  High dimension ({dim}) detected. Consider using tensor-based memory for better performance.")
        # Note: use_tensor_memory will be defined later, so we'll check after it's defined

    # Memory buffer capacities
    live_capacity = 256
    replay_capacity = 2048

    # Performance optimization for large simulations
    use_tensor_memory = False  # Set to True for better performance with large buffers

    # Performance warning for large simulations
    if n_initial_memory > 1000 or n_steps > 1000:
        if not use_tensor_memory:
            print("⚠️  Performance Warning: Large simulation detected.")
            print("   Consider setting use_tensor_memory=True for better performance.")
            print("   This will use tensor-based memory management instead of deque operations.")

    # Additional dimension-based performance recommendation
    if dim > 512 and not use_tensor_memory:
        print("   Recommendation: Set use_tensor_memory=True for dimensions > 512")

    # Alpha control settings
    alpha_mode = "adaptive"  # "cosine" or "adaptive"
    alpha_min, alpha_max = 0.1, 0.9
    cosine_period = 100

    # Replay sampling preference threshold for undirected mode
    alpha_replay_thresh = 0.4

    # Noise scaling method
    noise_scale = "sqrt"  # "linear", "sqrt", or "adaptive"

    # Novelty computation options
    deterministic_novelty_sampling = False  # For reproducible results in high-precision studies
    novelty_sampling_seed = 42  # Seed for deterministic novelty sampling

    # Initialize memory buffers
    buffers = MemoryBuffers(dim=dim, live_capacity=live_capacity, replay_capacity=replay_capacity,
                           use_tensor_memory=use_tensor_memory)
    buffers.initialize(n_initial_memory, replay_fraction=0.3)

    # Initialize alpha controller
    alpha_ctrl = AlphaController(
        mode=alpha_mode,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        period=cosine_period,
    )

    # Storage for logs
    novelty_log = []
    coherence_log = []
    creativity_log = []
    alpha_log = []
    mem_live_log, mem_replay_log, mem_total_log = [], [], []

    # Main simulation loop
    for step in range(n_steps):
        # Update alpha before sampling for adaptive mode to influence current step
        if step > 0:  # Use previous creativity for first step
            curr_alpha = alpha_ctrl.update(step - 1, creativity_log[-1])
        else:
            curr_alpha = alpha_ctrl.alpha

        # Sample two distinct vectors with replay preference when alpha is low
        x_i, x_j = buffers.sample_pair(curr_alpha, alpha_replay_thresh=alpha_replay_thresh)

        # Generate random noise vector
        noise = torch.randn(dim)

        # Reorganize: combine vectors with noise scaled by chosen method
        output = reorganize(x_i, x_j, curr_alpha, noise, noise_scale=noise_scale)

        # Compute metrics
        memory_tensor = buffers.full_memory()
        novelty = compute_novelty(output, memory_tensor,
                                 deterministic_sampling=deterministic_novelty_sampling,
                                 seed=novelty_sampling_seed)
        coherence = compute_coherence(output, x_i, x_j)
        creativity = novelty * coherence

        # Log metrics
        novelty_log.append(float(novelty.item()))
        coherence_log.append(float(coherence.item()))
        creativity_log.append(float(creativity.item()))
        alpha_log.append(curr_alpha)

        # Add output to memory buffers (live -> replay when overflowing)
        buffers.add(output)
        live_sz, replay_sz, total_sz = buffers.sizes()
        mem_live_log.append(live_sz)
        mem_replay_log.append(replay_sz)
        mem_total_log.append(total_sz)

        # Progress logging
        if (step + 1) % 20 == 0:
            print(
                f"Step {step + 1}/{n_steps}: "
                f"alpha={curr_alpha:.3f}, "
                f"novelty={novelty_log[-1]:.4f}, "
                f"coherence={coherence_log[-1]:.4f}, "
                f"creativity={creativity_log[-1]:.4f}, "
                f"mem_live={live_sz}, mem_replay={replay_sz}, total_mem={total_sz}"
            )

    # Plotting
    steps = list(range(1, n_steps + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Creativity
    axes[0, 0].plot(steps, creativity_log, label='Creativity', color='tab:purple',
                    linewidth=2, linestyle='-', marker='o', markersize=3, markevery=max(1, n_steps//20))
    axes[0, 0].set_title('Creativity over Steps', fontweight='bold')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Creativity')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc='upper right')

    # Novelty
    axes[0, 1].plot(steps, novelty_log, label='Novelty', color='tab:orange',
                    linewidth=2, linestyle='--', marker='s', markersize=3, markevery=max(1, n_steps//20))
    axes[0, 1].set_title('Novelty over Steps', fontweight='bold')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Novelty (normalized min distance)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc='upper right')

    # Coherence
    axes[1, 0].plot(steps, coherence_log, label='Coherence', color='tab:green',
                    linewidth=2, linestyle='-.', marker='^', markersize=3, markevery=max(1, n_steps//20))
    axes[1, 0].set_title('Coherence over Steps', fontweight='bold')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Coherence (clipped cosine similarity)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc='upper right')

    # Memory size and alpha with improved layout
    ax = axes[1, 1]
    ax.plot(steps, mem_total_log, label='Total Memory', color='tab:blue',
            linewidth=2, linestyle='-', marker='D', markersize=2, markevery=max(1, n_steps//15))
    ax.plot(steps, mem_live_log, label='Live Memory', color='tab:cyan',
            linewidth=1.5, linestyle=':', marker='o', markersize=2, markevery=max(1, n_steps//15), alpha=0.8)
    ax.plot(steps, mem_replay_log, label='Replay Memory', color='tab:red',
            linewidth=1.5, linestyle='-.', marker='s', markersize=2, markevery=max(1, n_steps//15), alpha=0.8)
    ax.set_title('Memory Growth and Alpha Control', fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Memory Size')
    ax.grid(True, alpha=0.3)

    # Add vertical lines for alpha thresholds to visualize exploration vs exploitation phases
    ax2 = ax.twinx()
    ax2.plot(steps, alpha_log, label='Alpha', color='black', linestyle='--',
             linewidth=2, marker='x', markersize=2, markevery=max(1, n_steps//10), alpha=0.7)
    ax2.axhline(y=alpha_replay_thresh, color='gray', linestyle=':', alpha=0.5,
                label=f'Replay Threshold ({alpha_replay_thresh})')
    ax2.set_ylabel('Alpha (Exploration ← → Exploitation)')
    ax2.set_ylim(0, 1)

    # Create separate legends for clarity
    # Memory legend
    memory_lines, memory_labels = ax.get_legend_handles_labels()
    ax.legend(memory_lines, memory_labels, loc='upper left', bbox_to_anchor=(0.02, 0.98))

    # Alpha legend
    alpha_lines, alpha_labels = ax2.get_legend_handles_labels()
    ax2.legend(alpha_lines, alpha_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    fig.suptitle('Creativity Simulation Metrics', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    plt.savefig('creativity_plot.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'creativity_plot.png'")
    print(f"Final memory sizes: live={mem_live_log[-1]}, replay={mem_replay_log[-1]}, total={mem_total_log[-1]}")
    print(f"Average creativity score: {sum(creativity_log) / len(creativity_log):.4f}")
    plt.show()


if __name__ == "__main__":
    main()
