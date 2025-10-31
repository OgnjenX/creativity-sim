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

from collections import deque
import math
from typing import Tuple, Optional

import torch
import matplotlib.pyplot as plt


def reorganize(x_i: torch.Tensor, x_j: torch.Tensor, alpha: float, noise: torch.Tensor) -> torch.Tensor:
    """
    Combine two input vectors with weighted averaging and scaled noise.
    
    Args:
        x_i: First input vector (tensor)
        x_j: Second input vector (tensor)
        alpha: Weighting factor for combination (float in [0, 1])
        noise: Random noise vector (tensor)
    
    Returns:
        Combined vector with noise scaled by (1 - alpha)
    """
    combined = alpha * x_i + (1 - alpha) * x_j
    result = combined + (1 - alpha) * noise
    return result


def compute_novelty(vector: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    """
    Compute novelty as the minimum Euclidean distance to all vectors in memory.
    """
    if memory.numel() == 0:
        # Return infinity to indicate maximal novelty when memory is empty
        return torch.tensor(float('inf'), dtype=vector.dtype)
    distances = torch.norm(memory - vector.unsqueeze(0), dim=1)
    return torch.min(distances)


def compute_coherence(vector: torch.Tensor, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
    """
    Compute coherence as the average cosine similarity to input vectors,
    clipped to [0, 1].
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
        return float(self._alpha)

    def update(self, step: int, creativity: float) -> float:
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
    """Manages live and replay memories with fixed capacities and biased sampling."""

    def __init__(self, dim: int, live_capacity: int = 256, replay_capacity: int = 2048):
        self.dim = dim
        self.live = deque(maxlen=live_capacity)
        self.replay = deque(maxlen=replay_capacity)

    def initialize(self, n_initial: int) -> None:
        init = torch.randn(n_initial, self.dim)
        for v in init:
            self.add(v)

    def add(self, vector: torch.Tensor) -> None:
        # If live is at capacity, capture the oldest element before it is dropped by the next append; push that to replay
        if len(self.live) == self.live.maxlen:
            oldest = self.live[0]
            # Move to replay; deque maxlen auto-drops the oldest if full
            self.replay.append(oldest)
        self.live.append(vector.detach())

    def full_memory(self) -> torch.Tensor:
        if len(self.live) == 0 and len(self.replay) == 0:
            return torch.empty(0, self.dim)
        all_vecs = list(self.replay) + list(self.live)
        return torch.stack(all_vecs, dim=0)

    def sizes(self) -> Tuple[int, int, int]:
        return len(self.live), len(self.replay), len(self.live) + len(self.replay)

    @staticmethod
    def _draw_from(buf: deque) -> int:
        # Return index into buffer
        return int(torch.randint(0, len(buf), (1,)).item())

    @staticmethod
    def _draw_distinct(buf: deque, avoid_idx: int, fallback_buf: Optional[deque] = None) -> torch.Tensor:
        """Draw a vector from buf with an index different from avoid_idx.
        If buf has size 1 and fallback_buf is provided, draw from fallback_buf.
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
            # Fallback: return the only available vector (should not happen due to assert)
            return buf[avoid_idx]

    def sample_pair(self, alpha: float, alpha_replay_thresh: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample two distinct vectors, preferentially from replay when alpha is low.

        Strategy: compute preference p toward replay when alpha <= threshold.
        Draw first sample accordingly; for second, try to draw from the same buffer
        while ensuring distinctness; fallback to the other buffer if needed.
        """
        total_mem = self.sizes()[2]
        assert total_mem >= 2, f"Not enough memory to sample two distinct vectors (current size: {total_mem}, required: 2)"
        have_replay = len(self.replay) > 0
        have_live = len(self.live) > 0

        # Preference for replay when alpha is small
        p_replay = 0.0
        if alpha_replay_thresh <= 0:
            raise ValueError("alpha_replay_thresh must be greater than zero for proper replay preference calculation.")
        if alpha <= alpha_replay_thresh and have_replay:
            # Linearly increase preference as alpha -> 0
            p_replay = (alpha_replay_thresh - alpha) / alpha_replay_thresh
            p_replay = float(min(max(p_replay, 0.0), 1.0))

        # Decide source for first draw
        use_replay_first = have_replay and (not have_live or torch.rand(()) < p_replay)
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
    """Main simulation loop for creativity exploration."""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Configuration
    dim = 16  # Dimension of latent vectors
    n_initial_memory = 100  # Initial number of memory vectors
    n_steps = 300  # Number of simulation steps

    # Memory buffer capacities
    live_capacity = 256
    replay_capacity = 2048

    # Alpha control settings
    alpha_mode = "adaptive"  # "cosine" or "adaptive"
    alpha_min, alpha_max = 0.1, 0.9
    cosine_period = 100

    # Replay sampling preference threshold for undirected mode
    alpha_replay_thresh = 0.4

    # Initialize memory buffers
    buffers = MemoryBuffers(dim=dim, live_capacity=live_capacity, replay_capacity=replay_capacity)
    buffers.initialize(n_initial_memory)

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
        # Sample two distinct vectors with replay preference when alpha is low
        # Use current alpha for sampling; if adaptive, use last known value
        curr_alpha = alpha_ctrl.alpha
        x_i, x_j = buffers.sample_pair(curr_alpha, alpha_replay_thresh=alpha_replay_thresh)

        # Generate random noise vector
        noise = torch.randn(dim)

        # Reorganize: combine vectors with noise scaled by (1 - alpha)
        output = reorganize(x_i, x_j, curr_alpha, noise)

        # Compute metrics
        memory_tensor = buffers.full_memory()
        novelty = compute_novelty(output, memory_tensor)
        coherence = compute_coherence(output, x_i, x_j)
        creativity = novelty * coherence

        # Update alpha after observing creativity (continuous control)
        curr_alpha = alpha_ctrl.update(step, float(creativity.item()))

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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Creativity
    axes[0, 0].plot(steps, creativity_log, label='Creativity', color='tab:purple')
    axes[0, 0].set_title('Creativity over Steps')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Creativity')
    axes[0, 0].grid(True, alpha=0.3)

    # Novelty
    axes[0, 1].plot(steps, novelty_log, label='Novelty', color='tab:orange')
    axes[0, 1].set_title('Novelty over Steps')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Novelty (min distance)')
    axes[0, 1].grid(True, alpha=0.3)

    # Coherence
    axes[1, 0].plot(steps, coherence_log, label='Coherence', color='tab:green')
    axes[1, 0].set_title('Coherence over Steps')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Coherence (clipped cos)')
    axes[1, 0].grid(True, alpha=0.3)

    # Memory size and alpha (optional overlay)
    ax = axes[1, 1]
    ax.plot(steps, mem_total_log, label='Memory Size (total)', color='tab:blue')
    ax.plot(steps, mem_live_log, label='Live Size', color='tab:cyan', alpha=0.7)
    ax.plot(steps, mem_replay_log, label='Replay Size', color='tab:red', alpha=0.7)
    ax.set_title('Memory Growth over Steps')
    ax.set_xlabel('Step')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(steps, alpha_log, label='Alpha', color='black', linestyle='--', alpha=0.6)
    ax2.set_ylabel('Alpha')

    # Legends
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].legend()
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    fig.suptitle('Creativity Simulation Metrics', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig('creativity_plot.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'creativity_plot.png'")
    print(f"Final memory sizes: live={mem_live_log[-1]}, replay={mem_replay_log[-1]}, total={mem_total_log[-1]}")
    print(f"Average creativity score: {sum(creativity_log) / len(creativity_log):.4f}")
    plt.show()


if __name__ == "__main__":
    main()
