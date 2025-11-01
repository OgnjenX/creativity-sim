# creativity-sim

A PyTorch-based simulation of creativity through vector combination and exploration.

## Overview

This simulation explores creativity by combining random latent vectors with varying degrees of exploration (controlled by alpha parameter) and measuring the novelty and coherence of the resulting combinations.

## Features

- **Memory System**: 100 random latent vectors (dimension 16) initialized as memory
- **Reorganize Function**: Combines two vectors with weighted averaging and scaled noise
- **Alpha Alternation**: Alpha alternates between 0.2 and 0.8 every 10 steps for varied exploration
- **Novelty Metric**: Measures minimum Euclidean distance to memory vectors
- **Coherence Metric**: Measures average cosine similarity to input vectors
- **Creativity Metric**: Product of novelty and coherence
- **Visualization**: Plots creativity score over 100 simulation steps

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python creativity_sim.py
```

This will:
1. Run the simulation for 100 steps
2. Print progress every 20 steps
3. Generate a plot saved as `creativity_plot.png`
4. Display final statistics

## How It Works

1. **Initialization**: Creates 100 random 16-dimensional vectors as initial memory
2. **Main Loop** (100 steps):
   - Select two random vectors from memory
   - Combine them using the reorganize function with current alpha value
   - Add noise scaled by (1-alpha) for exploration
   - Compute novelty (minimum distance to all memory vectors)
   - Compute coherence (average cosine similarity to input vectors)
   - Calculate creativity = novelty × coherence × competence
   - Append the new vector to memory
3. **Visualization**: Plot creativity scores over time, showing how alpha transitions affect exploration

## Parameters

- **dim**: 16 (dimension of latent vectors)
- **n_initial_memory**: 100 (initial number of memory vectors)
- **n_steps**: 100 (number of simulation steps)
- **alpha**: Alternates between 0.2 (exploration) and 0.8 (exploitation) every 10 steps

## Competence Metric (Reorganizational)

Creativity is extended to be “novel, coherent, and reorganizational.” The new
competence metric captures improvements in representational structure after each step:

- Diversity change: increase in average pairwise distance of memory (normalized)
- Prototype shift: magnitude of memory centroid change (normalized)

Competence is centered near 1.0 and bounded: competence ≈ 1 + 0.5 · tanh(Δstructure),
and is included multiplicatively: creativity = novelty × coherence × competence.
The plot now shows novelty, coherence, competence, and creativity over time, plus
memory growth/alpha and memory diversity.
