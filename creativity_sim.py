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
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.nn import functional as F

from alpha_controller import AlphaController
from memory import MemoryBuffers
from metrics import (
    reorganize,
    compute_novelty,
    compute_coherence,
    compute_competence,
    sample_memory_for_normalization,
    avg_pairwise_distance,
)

LEGEND_LOC_UPPER_RIGHT = "upper right"


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
    pair_type: str | None = None,
) -> None:
    ctx_str = f", ctx_pair={pair_type}" if pair_type is not None else ""
    print(
        f"Step {step}/{n_steps}: "
        f"alpha={curr_alpha:.3f}, "
        f"novelty={novelty:.4f}, "
        f"coherence={coherence:.4f}, "
        f"competence={competence:.4f}, "
        f"creativity={creativity:.4f}, "
        f"mem_live={live_sz}, mem_replay={replay_sz}, total_mem={total_sz}" + ctx_str
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
    context_pair_type_log: list[str] | None = None,
    context_avg_block: int | None = None,
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

    # Overlay binned averages of creativity for same vs cross-context pairs
    if context_pair_type_log is not None and len(context_pair_type_log) == len(steps):
        block = max(1, int(context_avg_block or 50))
        same_x, same_y, cross_x, cross_y = [], [], [], []
        for start in range(0, len(steps), block):
            end = min(len(steps), start + block)
            blk_steps = steps[start:end]
            blk_types = context_pair_type_log[start:end]
            blk_vals = creativity_log[start:end]
            same_vals = [v for v, t in zip(blk_vals, blk_types) if t == "same"]
            cross_vals = [v for v, t in zip(blk_vals, blk_types) if t == "cross"]
            x_pos = blk_steps[-1]
            if same_vals:
                same_x.append(x_pos)
                same_y.append(sum(same_vals) / len(same_vals))
            if cross_vals:
                cross_x.append(x_pos)
                cross_y.append(sum(cross_vals) / len(cross_vals))
        if same_x:
            axes[0, 0].plot(
                same_x, same_y, color="tab:blue", marker="o", linestyle="none", alpha=0.8,
                label=f"Same-ctx avg ({block}-step)",
            )
        if cross_x:
            axes[0, 0].plot(
                cross_x, cross_y, color="tab:red", marker="s", linestyle="none", alpha=0.8,
                label=f"Cross-ctx avg ({block}-step)",
            )
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
    parser.set_defaults(enable_competence=None)
    # Parse only known to avoid conflicting with external launchers
    args, _ = parser.parse_known_args()

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

    return cfg


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
    for M in contexts:
        try:
            invM = torch.linalg.inv(M)
        except RuntimeError:
            invM = torch.pinverse(M)
        contexts_inv.append(invM)
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
    # Optionally clip novelty to avoid explosion
    novelty = torch.clamp(novelty, 0.0, 5.0)
    coherence = compute_coherence(output, x_i, x_j)

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
        # Sample contexts for parents with cross-context control
        n_ctx = config["n_contexts"]
        cross_prob = float(config["cross_context_prob"])
        if float(torch.rand(()).item()) < cross_prob:
            ctx_i = int(torch.randint(0, n_ctx, (1,)).item())
            # ensure different
            offset = int(torch.randint(1, n_ctx, (1,)).item())
            ctx_j = (ctx_i + offset) % n_ctx
            pair_type = "cross"
        else:
            ctx_i = int(torch.randint(0, n_ctx, (1,)).item())
            ctx_j = ctx_i
            pair_type = "same"

        contexts = config["contexts"]
        contexts_inv = config["contexts_inv"]
        # Apply context transformations before recombination
        x_i_ctx = contexts[ctx_i] @ x_i
        x_j_ctx = contexts[ctx_j] @ x_j
        noise = torch.randn(config["dim"])
        if extra_noise > 0:
            noise = noise + extra_noise * torch.randn(config["dim"])
        output_ctx = reorganize(
            x_i_ctx, x_j_ctx, alpha_effective, noise, noise_scale=config["noise_scale"]
        )
        # Project result back according to configured strategy
        if config.get("context_projection", "base") == "avg":
            M_avg = 0.5 * (contexts_inv[ctx_i] + contexts_inv[ctx_j])
            output = M_avg @ output_ctx
        else:
            # Default: project to base context (0)
            output = contexts_inv[0] @ output_ctx
        # Log context pair type for this output
        logs["context_pair_type_log"].append(pair_type)
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
                pair_type,
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
        logs["context_pair_type_log"],
        config.get("context_avg_block", 50),
    )
    print(
        f"Final memory sizes: live={logs['mem_live_log'][-1]}, "
        f"replay={logs['mem_replay_log'][-1]}, total={logs['mem_total_log'][-1]}"
    )
    avg_creativity = sum(logs["creativity_log"]) / len(logs["creativity_log"]) if logs["creativity_log"] else 0.0
    print(f"Average creativity score: {avg_creativity:.4f}")
    if logs["creativity_same"]:
        print(f"Same-context avg creativity: {sum(logs['creativity_same']) / len(logs['creativity_same']):.4f}")
    if logs["creativity_cross"]:
        print(
            f"Cross-context avg creativity: {sum(logs['creativity_cross']) / len(logs['creativity_cross']):.4f}"
        )
    plt.show()


if __name__ == "__main__":
    main()
