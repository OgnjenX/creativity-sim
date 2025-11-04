"""Structured representation mode for creativity simulation.

Simulates cortical feature maps (e.g., shape, color, texture) and hippocampal
recombination of feature associations. Inspired by the Thousand Brains Theory:
each feature corresponds to a cortical submap; creative recombination emerges
from novel yet coherent cross-map combinations.
"""

import torch


# Default discrete feature spaces for structured objects
FEATURE_SPACES = {
    "shape": ["circle", "square", "triangle"],
    "color": ["red", "green", "blue"],
    "size": ["small", "medium", "large"],
    "texture": ["smooth", "rough"],
    "brightness": ["dim", "bright"],
}


class StructuredMemoryEncoder(torch.nn.Module):
    """Encode dictionaries of discrete features into a continuous latent vector.

    Each feature name has an embedding table. Objects are encoded by concatenating
    the per-feature embeddings into a single vector.
    """

    def __init__(self, feature_spaces: dict[str, list[str]], dim: int = 8) -> None:
        super().__init__()
        self.embeddings = torch.nn.ModuleDict(
            {name: torch.nn.Embedding(len(values), dim) for name, values in feature_spaces.items()}
        )
        self.feature_spaces = feature_spaces
        self.dim = dim

    def forward(self, obj_dict: dict[str, int]) -> torch.Tensor:
        """Forward pass: encodes object dict to continuous latent vector."""
        return self.encode(obj_dict)

    def encode(self, obj_dict: dict[str, int]) -> torch.Tensor:
        vecs: list[torch.Tensor] = []
        for name, val in obj_dict.items():
            idx = torch.tensor(val, dtype=torch.long)
            vecs.append(self.embeddings[name](idx))
        return torch.cat(vecs)

    def decode(self, obj_dict: dict[str, int]) -> dict[str, str]:
        return {k: self.feature_spaces[k][v] for k, v in obj_dict.items()}


def random_object(feature_spaces: dict[str, list[str]]) -> dict[str, int]:
    """Sample a random object as a dict of feature indices."""
    return {k: int(torch.randint(0, len(v), (1,)).item()) for k, v in feature_spaces.items()}


def sample_structured_memories(
    n: int, encoder: StructuredMemoryEncoder
) -> tuple[list[dict[str, int]], torch.Tensor]:
    """Generate ``n`` random structured objects and their encoded vectors."""
    objects: list[dict[str, int]] = []
    vecs: list[torch.Tensor] = []
    for _ in range(n):
        obj = random_object(encoder.feature_spaces)
        objects.append(obj)
        vecs.append(encoder.encode(obj))
    return objects, (torch.stack(vecs) if vecs else torch.empty(0, 0))


def recombine_objects(
    obj1: dict[str, int],
    obj2: dict[str, int],
    feature_spaces: dict[str, list[str]],
    alpha: float = 0.5,
) -> dict[str, int]:
    """Feature-wise recombination between two parent objects.

    For each feature, choose obj1's value with probability ``alpha``, else obj2's.
    """
    child: dict[str, int] = {}
    for k in feature_spaces.keys():
        child[k] = obj1[k] if float(torch.rand(1).item()) < float(alpha) else obj2[k]
    return child


def _semantic_pairs(feature_spaces: dict[str, list[str]]):
    names = list(feature_spaces.keys())
    for i, name_i in enumerate(names):
        for j in range(i + 1, len(names)):
            yield name_i, names[j]


def update_semantic_counts(
    counts: dict[tuple[tuple[str, int], tuple[str, int]], int],
    obj: dict[str, int],
    feature_spaces: dict[str, list[str]],
) -> None:
    """Update co-occurrence counts for all feature-value pairs present in ``obj``."""
    for a, b in _semantic_pairs(feature_spaces):
        key = ((a, obj[a]), (b, obj[b]))
        counts[key] = counts.get(key, 0) + 1


def compute_semantic_coherence(
    obj: dict[str, int],
    counts: dict[tuple[tuple[str, int], tuple[str, int]], int],
    seen: int,
    feature_spaces: dict[str, list[str]],
) -> float:
    """Average co-occurrence probability of the object's feature pairs.

    Simple estimator: for each feature pair (A=v_a, B=v_b), use count/seen.
    Returns value in [0, 1].
    """
    if seen <= 0:
        return 0.5  # neutral prior
    probs: list[float] = []
    for a, b in _semantic_pairs(feature_spaces):
        key = ((a, obj[a]), (b, obj[b]))
        probs.append(float(counts.get(key, 0)) / float(seen))
    return float(sum(probs) / len(probs)) if probs else 0.5
