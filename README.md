# TRM Path Planning

Tiny Recursive Model (TRM) for 2D path planning on occupancy grids.

The model learns the optimal action policy (**N / S / E / W / stay**) at every free
cell of a 2D occupancy grid, given a start and a goal. Ground truth comes from
BFS/Dijkstra during data generation. The recursive, weight-shared architecture gives
the effective depth of a much deeper network — `T × (n+1) × num_layers` layer
applications per forward — at the parameter cost of a 2-layer one, making it suitable
for edge / robotics deployment.

The recommended config (`trm_1m`) is a **~1.2M-parameter** model that reaches
**100% path-success** on 26×26 grids at 25% obstacle density (see `results.md`).

Architecture reference: *Less is More: Recursive Reasoning with Tiny Networks*
(https://arxiv.org/pdf/2510.04871)

## How it works

The model keeps three tensors, each `(B, L, dim)` where `L = grid_size²`:

- `x` — the embedded input grid (fixed; every step sees it)
- `y` — the answer, decoded into per-cell action logits
- `z` — a reasoning scratchpad, recursed many times but never decoded directly

One **supervision step** (`T` of them, loss applied at each — *deep supervision*):

1. recurse `z` for `n` steps (the "thinking"), holding `y` fixed
2. update `y` once from the freshly-recursed `z` (the "answer")
3. decode `y` into per-cell action logits

A small stack of weight-shared `TRMLayer`s (pre-norm self-attention with 2D RoPE +
SwiGLU FFN) does all the work. An optional **halt head** predicts how "done" the
answer is. Only the last recursion of each step carries gradients (the 1-step-gradient
trick), keeping memory and training stable.

## Project structure

```
configs/    yaml configs — one knob-set per experiment
dataset/    synthetic grid generation (solver + builder) and parquet loading
models/     TRM (layers, attention, embeddings, architecture) and CNN baseline
utils/      metrics, rerun visualization, csv logging, EMA
scripts/    evaluation and visualization entry points
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

Tuned for an 8 GB-VRAM GPU (RTX 5060 Mobile): training uses bf16 autocast, which
roughly halves activation memory. Peak usage for `trm_1m` is well under 2 GB.

## Running the pipeline

### 1. Build the dataset

Generates random reachable grids, solves each for its optimal action field, and writes
`{train,val,test}` parquet splits to `data.data_dir` from the config.

```bash
uv run python -m dataset.build_dataset --config configs/trm_1m.yaml

# sanity-check the generated ground truth visually in rerun (logs 20 samples)
uv run python -m dataset.build_dataset --config configs/trm_1m.yaml --visualize 20
```

Grid size, obstacle density, and split sizes all come from the config's `data:` block.

### 2. Train

```bash
uv run python pretrain.py --config configs/trm_1m.yaml
```

Each epoch trains over the data, evaluates on the validation split using the EMA
weights, and saves the best checkpoint (by path-success rate) to
`logging.checkpoint_dir/best.pt`. If `logging.viz_samples > 0`, a fixed set of val
samples is logged to rerun each epoch so you can watch predicted paths converge.

Outputs:
- `checkpoints/<run>/best.pt` — EMA weights + the config used + epoch
- `logs/<run>/train_metrics.csv` — per-step `loss`, `train_acc`
- `logs/<run>/val_metrics.csv` — per-epoch `success_rate`, `optimality_ratio`, `per_cell_acc`
- `logs/<run>/train.rrd` — rerun recording of the live metric graphs + path viz (only
  when `viz_samples > 0`); reopen anytime with `rerun logs/<run>/train.rrd`

### 3. Evaluate & visualize

Runs the trained checkpoint on the held-out **test** split and logs predicted vs.
optimal paths to rerun (green = optimal, purple = predicted, with SUCCESS/FAILED labels).

```bash
uv run python -m scripts.eval_viz --config configs/trm_1m.yaml
uv run python -m scripts.eval_viz --config configs/trm_1m.yaml --n 50
uv run python -m scripts.eval_viz --config configs/trm_1m.yaml --checkpoint path/to/best.pt
```

The architecture is rebuilt from the config stored *inside* the checkpoint, so it
always matches the trained weights.

### 4. Baselines (A* + CNN)

To put the TRM in context, compare it against a classical planner and two
parameter-matched CNNs (`cnn_shallow`, `cnn_deep`). Train the CNNs on the same budget
as `trm_1m`, then run the two comparison scripts:

```bash
# train the param-matched CNN baselines (~1.2M each, trm_1m budget)
uv run python pretrain.py --config configs/cnn_shallow.yaml
uv run python pretrain.py --config configs/cnn_deep.yaml

# accuracy on the test split (A* oracle vs TRM vs CNNs)
uv run python -m scripts.compare_baselines

# latency / throughput across batch sizes and devices (bf16; add --compile to fuse)
uv run python -m scripts.benchmark_latency

# out-of-distribution sweeps over grid size and obstacle density
uv run python -m scripts.eval_ood
```

- `dataset/astar.py` — a true A* planner (Manhattan heuristic, 4-connected, optimal by
  construction); the latency oracle for the classical comparison.
- `scripts/compare_baselines.py` — success / optimality / per-cell-acc on the test split,
  reusing the training-time evaluator so the TRM numbers match its best checkpoint.
- `scripts/benchmark_latency.py` — per-grid latency and throughput for A*, BFS, both CNNs,
  and the TRM at batch sizes 1/8/64/512 on CPU and GPU.
- `scripts/eval_ood.py` — out-of-distribution sweeps: success / optimality / per-cell-acc on
  grid sizes and densities the models never trained on (the TRM runs at any size by rebuilding
  its relative 2D-RoPE tables). Test grids are generated on demand and cached under `data/ood/`.

See `results.md` for the recorded tables, the receptive-field analysis, and the OOD findings.

## Configs

| Config | Params | Notes |
|--------|--------|-------|
| `trm_1m.yaml` | ~1.2M | **Recommended.** dim 64, n 4. 100% success by epoch ~6. |
| `base.yaml` | ~4.8M | Larger reference model (dim 128, n 6). |
| `sanity.yaml` | small | Fast pipeline smoke-test (T 1, n 2, 10 epochs). |
| `ablation_n{1,3,12}.yaml` | varies | Sweep over reasoning recursions `n`. |
| `ablation_no_deepsup.yaml` | — | Deep supervision turned off. |
| `cnn_shallow.yaml` | ~1.2M | CNN baseline, 4 conv layers (9×9 receptive field). |
| `cnn_deep.yaml` | ~1.2M | CNN baseline, 13 conv layers (27×27 receptive field). |
| `cnn_baseline.yaml` | — | Original plain CNN baseline (`arch: cnn`). |

Key knobs: `model.{dim, num_heads, num_layers, T, n, halting}` and
`training.{batch_size, lr, num_epochs, halt_loss_weight, deep_supervision, ema_decay}`.

> **2D RoPE constraint:** `head_dim = 3*dim / num_heads` must be divisible by 4. If you
> change `dim` or `num_heads`, keep this satisfied (e.g. dim 64 / heads 4 → head_dim 48 ✓).

## Metrics

- **success_rate** — fraction of grids whose rolled-out predicted policy reaches the goal
- **optimality_ratio** — predicted path length / shortest length, over successful grids (1.0 = optimal)
- **per_cell_acc** — fraction of cells with the correct predicted action

## Logging

Training logs to three places, all under `logs/<run>/`:

- **CSV** — `train_metrics.csv` (per-step `loss`, `train_acc`) and `val_metrics.csv`
  (per-epoch `success_rate`, `optimality_ratio`, `per_cell_acc`) for quick scripting/plotting.
- **Rerun live graphs** — when `viz_samples > 0`, scalar metrics stream to the rerun viewer
  as time-series so you can watch training in real time. Training curves (`train/loss`,
  `train/acc`, `train/lr`) plot against a global-step timeline; validation curves
  (`val/success_rate`, `val/optimality_ratio`, `val/per_cell_acc`) plot against an epoch
  timeline. The same fixed val samples' predicted-vs-optimal paths are logged each epoch.
- **Persistent `.rrd` recording** — the live stream is *also* saved to `logs/<run>/train.rrd`,
  so the graphs survive after the run ends. Reopen them anytime (no retraining) with:

  ```bash
  rerun logs/<run>/train.rrd     # e.g. rerun logs/trm_1m/train.rrd
  ```

  In the viewer, scrub the **step** timeline for the training curves and the **epoch**
  timeline for the validation curves.

### Plotting metrics offline (matplotlib)

As a local, no-server alternative to the rerun viewer, `scripts/plot_metrics.py` renders
the CSV logs into static figures — the **whole** training/val history at once (no
follow-mode scrolling), saved as a PNG you can drop straight into `results.md`. Pass
several runs to overlay and compare them (the one thing the live rerun view can't do well).

```bash
# single run (derive log_dir from a config, or point at it directly)
uv run python -m scripts.plot_metrics --config configs/trm_1m.yaml
uv run python -m scripts.plot_metrics --log-dir logs/trm_1m

# overlay multiple runs for comparison
uv run python -m scripts.plot_metrics --log-dir logs/trm_1m --log-dir logs/base

# custom output path (default: <log_dir>/metrics.png)
uv run python -m scripts.plot_metrics --config configs/trm_1m.yaml --out figs/trm_1m.png
```

Panels: train loss, train accuracy, val success-rate, optimality-ratio, per-cell-accuracy.
(Learning rate is only streamed to rerun, not the CSVs, so it isn't plotted here.)

Weights & Biases integration is planned but not yet wired in.

## Known limitations

The released model is trained only on **26×26 grids at 25% obstacle density**. Out-of-
distribution behaviour is now measured (`scripts/eval_ood.py`, see `results.md`): it
generalizes across **obstacle density** essentially for free (≥0.99 success from 0.10 to
0.40) and **extrapolates across grid size** markedly better than a param-matched CNN,
holding up to ~1.5× the trained side (0.94 success at 32×32) before degrading at 40×40
(0.52). Recovering large-grid success (e.g. via mixed-size training or more recursion) is
the main open thread.
