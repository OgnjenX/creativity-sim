"""Visualization and plotting functions for creativity simulation.

Handles creation and saving of plots for simulation metrics including
creativity, novelty, coherence, competence, and memory dynamics.
"""

import matplotlib.pyplot as plt

LEGEND_LOC_UPPER_RIGHT = "upper right"
COLOR_TAB_BLUE = "tab:blue"
COLOR_TAB_RED = "tab:red"


def plot_context_pair_averages(
    axes, steps, creativity_log, context_pair_type_log, context_avg_block
):
    """Plot binned averages of creativity for same vs cross-context pairs."""
    if context_pair_type_log is None or len(context_pair_type_log) != len(steps):
        return

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
            same_x,
            same_y,
            color=COLOR_TAB_BLUE,
            marker="o",
            linestyle="none",
            alpha=0.8,
            label=f"Same-ctx avg ({block}-step)",
        )
    if cross_x:
        axes[0, 0].plot(
            cross_x,
            cross_y,
            color=COLOR_TAB_RED,
            marker="s",
            linestyle="none",
            alpha=0.8,
            label=f"Cross-ctx avg ({block}-step)",
        )
    axes[0, 0].legend(loc=LEGEND_LOC_UPPER_RIGHT)


def log_progress(
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
    """Log progress of simulation step."""
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


def plot_metrics(
    steps,
    creativity_log,
    novelty_log,
    diversity_log,
    competence_log,
    coherence_log,
    alpha_log,
    alpha_replay_thresh,
    memory_logs: dict,
    plot_opts: dict | None = None,
):
    """Create and save comprehensive metrics plot."""
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
    if plot_opts is None:
        plot_opts = {}
    context_pair_type_log = plot_opts.get("context_pair_type_log")
    context_avg_block = plot_opts.get("context_avg_block", 50)
    plot_context_pair_averages(
        axes, steps, creativity_log, context_pair_type_log, context_avg_block
    )

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
        memory_logs["mem_total_log"],
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
        memory_logs["mem_live_log"],
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
        memory_logs["mem_replay_log"],
        label="Replay Memory",
        color=COLOR_TAB_RED,
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
    pulse_steps = plot_opts.get("pulse_steps")
    if pulse_steps:
        for s in pulse_steps:
            for row in range(2):
                for col in range(3):
                    axes[row, col].axvline(x=s, color=COLOR_TAB_RED, alpha=0.2, linestyle=":")
    fig.suptitle("Creativity Simulation Metrics", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    plt.savefig("creativity_plot.png", dpi=300, bbox_inches="tight")
    print("\nPlot saved as 'creativity_plot.png'")
