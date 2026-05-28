# TRM Path Planning

Tiny Recursive Model for 2D path planning on occupancy grids.

A 5–10M parameter model that learns the optimal action policy (N / S / E / W / stay)
at every free cell of a 2D occupancy grid, given a start and goal. Ground truth comes
from BFS/Dijkstra during data generation. The recursive, weight-shared architecture
gives the effective depth of a ~42-layer network at the parameter cost of a 2-layer one,
making it suitable for edge robotics deployment.

Architecture reference: *Less is More: Recursive Reasoning with Tiny Networks*
(https://arxiv.org/pdf/2510.04871)

## Project structure

```
configs/    yaml configs — one knob-set per experiment
dataset/    synthetic grid generation (solver + builder) and parquet loading
models/     TRM (layers, embeddings, architecture) and CNN baseline
utils/      EMA, metrics, rerun visualization, csv logging
scripts/    evaluation and benchmarking entry points
pretrain.py main training entry
results.md  hand-written experiment log
```

## Setup

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# create the environment from the lockfile
uv sync

# verify CUDA
uv run python -c "import torch; print('cuda:', torch.cuda.is_available())"
```
