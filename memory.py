"""Memory buffer management for creativity simulation.

This module provides the MemoryBuffers class for managing live and replay
memories with fixed capacities and biased sampling strategies.
"""

from collections import deque
from typing import Optional, Any, Tuple

import torch
from torch import Tensor

from metrics import knn_distance


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
        _, _ = knn_distance(v, mem, k=self.knn_k, normalize=True)
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
