"""Creativity Simulation using PyTorch

This script simulates creativity by combining latent vectors with varying degrees of
exploration (controlled by alpha) and measuring the novelty and coherence of the
resulting combinations. It includes:

- Distinct sampling of memory vectors for recombination
- Separate live and replay memories with preferential sampling in undirected mode
- Dynamic/continuous alpha control (cosine schedule or adaptive controller)
- Refined coherence metric (cosine similarity clipped to [0,1])
- Fixed-size memory buffers and tracking of memory growth
- Enhanced plotting of novelty, coherence, competence, creativity (and optional memory size)

Structured representation mode:
Simulates cortical feature maps (e.g., shape, color, texture) and hippocampal
recombination of feature associations. Inspired by the Thousand Brains Theory:
each feature corresponds to a cortical submap; creative recombination emerges
from novel yet coherent cross-map combinations.
"""

import argparse
import os
from typing import cast

from matplotlib import pyplot as plt
import torch
from torch import Tensor
from torch.nn import functional as F

from alpha_controller import AlphaController
from memory import MemoryBuffers
from metrics import (
    avg_pairwise_distance,
    compute_coherence,
    compute_competence,
    compute_novelty,
    reorganize,
    sample_memory_for_normalization,
)
from structured_representation import (
    FEATURE_SPACES,
    StructuredMemoryEncoder,
    random_object,
    sample_structured_memories,
    recombine_objects,
    compute_semantic_coherence,
    update_semantic_counts,
)

from visualization import (
    log_progress,
    plot_metrics,
)


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
    # Parse minimal CLI flags first to allow overrides
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--enable-competence",
        dest="enable_competence",
        action="store_true",
        help="Enable reorganizational competence metric",
    )
    parser.add_argument(
        "--no-enable-competence",
        dest="enable_competence",
        action="store_false",
        help="Disable reorganizational competence metric",
    )
    parser.add_argument(
        "--latent_mode",
        type=str,
        choices=["random", "structured"],
        default=os.getenv("LATENT_MODE", "random"),
        help="Latent representation mode: random vectors or structured objects",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=int(os.getenv("EMBED_DIM", "8")),
        help="Per-feature embedding dimension for structured mode",
    )
    parser.add_argument(
        "--semantic_coherence",
        action="store_true",
        help="Enable hybrid semantic coherence from feature co-occurrence",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of simulation steps (for quick tests)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging of structured objects and semantic stats",
    )
    parser.set_defaults(enable_competence=None)
    # Parse only known to avoid conflicting with external launchers
    args, _ = parser.parse_known_args()

    dim = 16  # Dimension of latent vectors (try 32, 64, 128 for richer dynamics)
    n_initial_memory = 100  # Initial number of memory vectors
    n_steps = args.epochs if args.epochs is not None else 300  # Number of simulation steps

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

    # Context / frame switching configuration
    n_contexts = 3
    cross_context_prob = 0.3
    context_transform_strength = 0.05  # scaling factor for random transformations (reduced)
    context_projection = "base"  # "base" or "avg" (average of parent frames)
    context_avg_block = 50  # steps per averaging block for plotting

    # Stagnation-triggered exploration pulse
    creativity_pulse = True
    pulse_window = 30
    pulse_drop_tol = 0.05  # trigger if drop vs. prior window exceeds this
    pulse_steps = 6
    pulse_noise_gain = 1.0  # extra noise magnitude added during pulse
    pulse_alpha_drop = 0.15  # temporarily decrease alpha by this amount

    # Validation
    _validate_dimensions(dim)
    _warn_large_simulation(n_initial_memory, n_steps, use_tensor_memory, dim)

    cfg = {
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
        # Representation mode
        "latent_mode": args.latent_mode,
        "embed_dim": int(args.embed_dim),
        # Contexts
        "n_contexts": n_contexts,
        "cross_context_prob": cross_context_prob,
        "context_transform_strength": context_transform_strength,
        "context_projection": context_projection,
        "context_avg_block": context_avg_block,
        "creativity_pulse": creativity_pulse,
        "pulse_window": pulse_window,
        "pulse_drop_tol": pulse_drop_tol,
        "pulse_steps": pulse_steps,
        "pulse_noise_gain": pulse_noise_gain,
        "pulse_alpha_drop": pulse_alpha_drop,
    }

    # Feature toggles (default True), allow ENV and CLI override
    enable_comp_default = True
    env_override = os.getenv("ENABLE_COMPETENCE")
    if env_override is not None:
        try:
            enable_comp_default = bool(int(env_override))
        except ValueError:
            enable_comp_default = env_override.lower() in {"true", "t", "yes", "y", "on"}
    if args.enable_competence is not None:
        cfg["enable_competence"] = bool(args.enable_competence)
    else:
        cfg["enable_competence"] = enable_comp_default

    # Semantic coherence toggle
    cfg["semantic_coherence"] = bool(args.semantic_coherence)
    cfg["seed"] = int(args.seed)
    cfg["verbose"] = bool(args.verbose)

    # If using structured mode, override dim based on feature spaces and embed_dim
    if cfg["latent_mode"] == "structured":
        feature_spaces = FEATURE_SPACES
        n_feats = len(feature_spaces)
        embed_dim = cfg["embed_dim"]
        cfg["dim"] = n_feats * embed_dim
        cfg["feature_spaces"] = feature_spaces

    # Safe defaults for semantic stats regardless of flag
    cfg.setdefault("semantic_counts", {})
    cfg.setdefault("semantic_seen", 0)

    return cfg


def _initialize_components(config):
    buffers = MemoryBuffers(
        dim=config["dim"],
        live_capacity=config["live_capacity"],
        replay_capacity=config["replay_capacity"],
        use_tensor_memory=config["use_tensor_memory"],
    )
    if config.get("latent_mode") == "structured":
        # Initialize structured encoder and populate memory with encoded objects
        encoder = StructuredMemoryEncoder(
            config.get("feature_spaces", FEATURE_SPACES), dim=config.get("embed_dim", 8)
        )
        objects, vecs = sample_structured_memories(config["n_initial_memory"], encoder)
        for v in vecs:
            buffers.add(v)
        config["encoder"] = encoder
        config["structured_objects"] = objects
        if config.get("semantic_coherence", False):
            counts: dict[tuple[tuple[str, int], tuple[str, int]], int] = {}
            for obj in objects:
                update_semantic_counts(counts, obj, config["feature_spaces"])
            config["semantic_counts"] = counts
            config["semantic_seen"] = len(objects)
    else:
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
    sample_for_div = sample_memory_for_normalization(
        init_memory, max_samples=200, deterministic=False, seed=None
    )
    config["baseline_diversity"] = float(avg_pairwise_distance(sample_for_div).item())

    # Initialize context transformation matrices (near-identity random transforms)
    dim = config["dim"]
    n_contexts = config["n_contexts"]
    strength = config["context_transform_strength"]
    contexts = [torch.eye(dim) + strength * torch.randn(dim, dim) for _ in range(n_contexts)]
    # Normalize rows to stabilize scales across contexts
    for i in range(n_contexts):
        contexts[i] = F.normalize(contexts[i], dim=1)
    # Precompute inverses for back-projection (fallback to pinverse if singular)
    contexts_inv = []
    for matrix in contexts:
        try:
            inv_matrix = torch.linalg.inv(matrix)  # pylint: disable=not-callable
        except RuntimeError:
            inv_matrix = torch.pinverse(matrix)
        contexts_inv.append(inv_matrix)
    config["contexts"] = contexts
    config["contexts_inv"] = contexts_inv

    return buffers, alpha_ctrl


def _initialize_logs():
    logs = {
        "novelty_log": [],
        "diversity_log": [],
        "competence_log": [],
        "coherence_log": [],
        "creativity_log": [],
        # Context pairing logs
        "context_pair_type_log": [],  # "same" | "cross"
        "creativity_same": [],
        "creativity_cross": [],
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


def _compute_and_log_metrics(
    output: Tensor,
    buffers: MemoryBuffers,
    config: dict,
    x_i: Tensor,
    x_j: Tensor,
    alpha_effective: float,
    logs: dict,
    *,
    child_obj: dict[str, int] | None = None,
):
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
    # Optionally clip novelty to avoid explosion
    novelty = torch.clamp(novelty, 0.0, 5.0)
    vec_coh_t = compute_coherence(output, x_i, x_j)
    # Optional semantic coherence (structured mode)
    if child_obj is not None and bool(config.get("semantic_coherence", False)):
        sem = compute_semantic_coherence(
            child_obj,
            config.get("semantic_counts", {}),
            int(config.get("semantic_seen", 0)),
            config.get("feature_spaces", FEATURE_SPACES),
        )
        coherence_val = 0.5 * (float(vec_coh_t.item()) + float(sem))
        coherence = torch.tensor(coherence_val, dtype=vec_coh_t.dtype)
    else:
        coherence = vec_coh_t

    # Update memory with new output
    # Normalize output vector before storing in memory
    output = output / (output.norm() + 1e-8)
    buffers.add(output)

    # Post-update memory for competence and diversity
    memory_after = buffers.full_memory()
    if config.get("enable_competence", True):
        competence, new_centroid, diversity_norm = compute_competence(
            logs.get("prev_centroid"),
            logs.get("prev_diversity_norm"),
            memory_after,
            config.get("baseline_diversity", 0.0),
        )
        # Update competence state
        logs["prev_centroid"] = new_centroid
        logs["prev_diversity_norm"] = diversity_norm
    else:
        competence = 1.0
    creativity = novelty * coherence * competence

    sample_for_div = sample_memory_for_normalization(
        memory_after, max_samples=200, deterministic=False, seed=None
    )
    diversity = float(avg_pairwise_distance(sample_for_div).item())

    # Log metrics
    logs["novelty_log"].append(float(novelty.item()))
    logs["competence_log"].append(float(competence))
    logs["diversity_log"].append(diversity)
    logs["coherence_log"].append(float(coherence.item()))
    logs["creativity_log"].append(float(creativity.item()))
    logs["alpha_log"].append(alpha_effective)

    # Memory sizes
    live_sz, replay_sz, total_sz = buffers.sizes()
    logs["mem_live_log"].append(live_sz)
    logs["mem_replay_log"].append(replay_sz)
    logs["mem_total_log"].append(total_sz)
    # Split creativity by last recorded context pair type (if present)
    if logs.get("context_pair_type_log"):
        last_type = logs["context_pair_type_log"][-1]
        if last_type == "same":
            logs["creativity_same"].append(float(creativity.item()))
        elif last_type == "cross":
            logs["creativity_cross"].append(float(creativity.item()))


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


def _sample_and_apply_contexts(config, x_i, x_j):
    """Sample contexts and apply transformations to input vectors."""
    n_ctx = config["n_contexts"]
    cross_prob = float(config["cross_context_prob"])

    if float(torch.rand(()).item()) < cross_prob:
        ctx_i = int(torch.randint(0, n_ctx, (1,)).item())
        offset = int(torch.randint(1, n_ctx, (1,)).item())
        ctx_j = (ctx_i + offset) % n_ctx
        pair_type = "cross"
    else:
        ctx_i = int(torch.randint(0, n_ctx, (1,)).item())
        ctx_j = ctx_i
        pair_type = "same"

    contexts = config["contexts"]
    x_i_ctx = contexts[ctx_i] @ x_i
    x_j_ctx = contexts[ctx_j] @ x_j

    return ctx_i, ctx_j, pair_type, x_i_ctx, x_j_ctx


def _project_output_back(config, output_ctx, ctx_i, ctx_j, contexts_inv):
    """Project reorganized output back to base context."""
    if config.get("context_projection", "base") == "avg":
        matrix_avg = 0.5 * (contexts_inv[ctx_i] + contexts_inv[ctx_j])
        return matrix_avg @ output_ctx
    else:
        return contexts_inv[0] @ output_ctx


def _process_structured_mode_step(config, buffers, logs, alpha_effective):
    """Execute one step in structured (object-based) mode."""
    objects: list[dict[str, int]] = config.get("structured_objects", [])
    encoder: StructuredMemoryEncoder = cast(StructuredMemoryEncoder, config.get("encoder"))

    # Select or generate parent objects
    if len(objects) < 2:
        obj_a = random_object(config["feature_spaces"])
        obj_b = random_object(config["feature_spaces"])
    else:
        idx_a = int(torch.randint(0, len(objects), (1,)).item())
        idx_b = int(torch.randint(0, len(objects), (1,)).item())
        while idx_b == idx_a and len(objects) > 1:
            idx_b = int(torch.randint(0, len(objects), (1,)).item())
        obj_a = objects[idx_a]
        obj_b = objects[idx_b]

    x_i = encoder.encode(obj_a)
    x_j = encoder.encode(obj_b)
    child_obj = recombine_objects(obj_a, obj_b, config["feature_spaces"], alpha=alpha_effective)
    output = encoder.encode(child_obj)

    # Update semantic stats and object memory
    if config.get("semantic_coherence", False):
        counts = config.get("semantic_counts", {})
        update_semantic_counts(counts, child_obj, config["feature_spaces"])
        config["semantic_counts"] = counts
        config["semantic_seen"] = int(config.get("semantic_seen", 0)) + 1

    objects.append(child_obj)
    config["structured_objects"] = objects
    logs["context_pair_type_log"].append(None)

    _compute_and_log_metrics(
        output, buffers, config, x_i, x_j, alpha_effective, logs, child_obj=child_obj
    )


def _process_random_mode_step(config, buffers, logs, alpha_effective, extra_noise):
    """Execute one step in random (vector-based) mode with context transformations."""
    x_i, x_j = _select_pair_or_fallback(
        buffers,
        alpha_effective,
        config["alpha_replay_thresh"],
        config["dim"],
        config["far_pair_prob"],
    )

    # Sample contexts and apply transformations
    ctx_i, ctx_j, pair_type, x_i_ctx, x_j_ctx = _sample_and_apply_contexts(config, x_i, x_j)
    contexts_inv = config["contexts_inv"]

    # Generate noise with pulse modulation
    noise = torch.randn(config["dim"])
    if extra_noise > 0:
        noise = noise + extra_noise * torch.randn(config["dim"])

    output_ctx = reorganize(
        x_i_ctx, x_j_ctx, alpha_effective, noise, noise_scale=config["noise_scale"]
    )

    # Project result back according to configured strategy
    output = _project_output_back(config, output_ctx, ctx_i, ctx_j, contexts_inv)
    logs["context_pair_type_log"].append(pair_type)

    _compute_and_log_metrics(output, buffers, config, x_i, x_j, alpha_effective, logs)


def _log_step_progress(step, config, logs):
    """Log progress at periodic intervals, with structured example decoding."""
    pair_type = logs["context_pair_type_log"][-1] if logs.get("context_pair_type_log") else None
    log_progress(
        step + 1,
        config["n_steps"],
        logs["alpha_log"][-1],
        logs["novelty_log"][-1],
        logs["coherence_log"][-1],
        logs["competence_log"][-1] if logs["competence_log"] else 1.0,
        logs["creativity_log"][-1],
        logs["mem_live_log"][-1],
        logs["mem_replay_log"][-1],
        logs["mem_total_log"][-1],
        pair_type,
    )
    if config.get("latent_mode") == "structured":
        encoder: StructuredMemoryEncoder = config["encoder"]
        objs = config.get("structured_objects", [])
        if objs:
            if hasattr(encoder, "decode"):
                decoded = encoder.decode(objs[-1])  # type: ignore[attr-defined]
            else:
                decoded = ", ".join(f"{k}:{v}" for k, v in objs[-1].items())
            print(f"Epoch {step + 1}: Example creative object -> {decoded}")
            if config.get("verbose") and config.get("semantic_coherence", False):
                sem = compute_semantic_coherence(
                    objs[-1],
                    config.get("semantic_counts", {}),
                    int(config.get("semantic_seen", 0)),
                    config.get("feature_spaces", FEATURE_SPACES),
                )
                print(f"  Semantic coherence (co-occurrence): {sem:.3f}")


def _finalize_simulation(config, logs):
    """Prepare final reporting and plot results."""
    steps = list(range(1, config["n_steps"] + 1))
    memory_logs = {
        "mem_live_log": logs["mem_live_log"],
        "mem_replay_log": logs["mem_replay_log"],
        "mem_total_log": logs["mem_total_log"],
    }
    plot_opts = {
        "pulse_steps": logs["pulse_markers"],
        "context_pair_type_log": logs["context_pair_type_log"],
        "context_avg_block": config.get("context_avg_block", 50),
    }

    # Use external visualization API; if signature mismatches, fallback is handled by that module
    plot_metrics(
        steps,
        logs["creativity_log"],
        logs["novelty_log"],
        logs["diversity_log"],
        logs["competence_log"],
        logs["coherence_log"],
        logs["alpha_log"],
        config["alpha_replay_thresh"],
        memory_logs,
        plot_opts,
    )

    # Print summary statistics
    print(
        f"Final memory sizes: live={logs['mem_live_log'][-1]}, "
        f"replay={logs['mem_replay_log'][-1]}, total={logs['mem_total_log'][-1]}"
    )

    if logs["creativity_log"]:
        avg_creativity = float(torch.tensor(logs["creativity_log"]).mean().item())
        print(f"Average creativity score: {avg_creativity:.4f}")

    if logs["creativity_same"]:
        same_avg = sum(logs["creativity_same"]) / len(logs["creativity_same"])
        print(f"Same-context avg creativity: {same_avg:.4f}")

    if logs["creativity_cross"]:
        cross_avg = sum(logs["creativity_cross"]) / len(logs["creativity_cross"])
        print(f"Cross-context avg creativity: {cross_avg:.4f}")

    plt.show()


def main():
    """Main simulation loop for creativity exploration.

    Configuration Tips:
    - dim: Higher dimensions (32, 64, 128) provide richer dynamics but may need more memory
    - n_steps: Longer simulations reveal more stable patterns in creativity metrics
    - use_tensor_memory: Recommended for dim > 512 or n_steps > 1000
    - noise_scale: 'sqrt' is default, 'adaptive' for balanced exploration
    """
    config = _setup_simulation_config()
    # Set random seed for reproducibility
    torch.manual_seed(config.get("seed", 42))
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

        # Execute step based on latent representation mode
        if config.get("latent_mode") == "structured":
            _process_structured_mode_step(config, buffers, logs, alpha_effective)
        else:
            _process_random_mode_step(config, buffers, logs, alpha_effective, extra_noise)

        # Check for creativity stagnation and trigger pulse if needed
        logs["pulse_counter"] = _check_and_trigger_pulse(
            config["creativity_pulse"],
            step,
            config["pulse_window"],
            logs["creativity_log"],
            config["pulse_drop_tol"],
            config["pulse_steps"],
            logs["pulse_markers"],
        )

        # Log progress periodically
        if (step + 1) % 20 == 0:
            _log_step_progress(step, config, logs)

    # Finalize and display results
    _finalize_simulation(config, logs)


if __name__ == "__main__":
    main()
