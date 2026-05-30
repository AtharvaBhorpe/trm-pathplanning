# Experiment Log

## Headline result

A **~1.2M-parameter** TRM (`configs/trm_1m.yaml`) reaches **100% path-success** with
**perfectly optimal paths** on 26×26 grids at 25% obstacle density, converging by
**epoch 6** (well under 10 minutes of training on an 8 GB RTX 5060 Mobile).

| Config | Params | Best val success | Optimality | Cell acc | Epochs to converge |
|--------|-------:|-----------------:|-----------:|---------:|-------------------:|
| `trm_1m` (dim 64, n 4) | 1,205,760 | **1.000** | **1.000** | **1.000** | ~6 |
| `base` (dim 128, n 6) | 4,819,968 | — (reference size) | — | — | — |

## Dataset

- **Grid size:** 26×26 (`L = 676` cells)
- **Obstacle density:** 25% (random, regenerated until start→goal is reachable)
- **Splits:** 40k train / 5k val / 10k test, disjoint seeds
- Ground-truth optimal action field per cell from BFS/Dijkstra (`dataset/solver.py`)

> ⚠️ The model is trained on a **single grid size and a single obstacle density**, so it
> may be fitted to 26×26 @ 25%. Generalization is untested — see *Open work* below.

## Convergence (final `trm_1m` run: dim 64, n 4, lr 3e-4, batch 32)

| Epoch | Success | Optimality | Cell acc |
|------:|--------:|-----------:|---------:|
| 1 | 0.023 | 1.023 | 0.723 |
| 2 | 0.888 | 1.008 | 0.965 |
| 3 | 0.990 | 1.001 | 0.992 |
| 4 | 0.997 | 1.000 | 0.996 |
| 5 | 0.999 | 1.000 | 0.998 |
| 6 | **1.000** | 1.000 | 0.999 |
| 10 | 1.000 | 1.000 | 1.000 |

Optimality is already ~1.02 at epoch 1 and hits a perfect 1.000 by epoch 4 — when the
model reaches the goal, it does so via a shortest path almost immediately; the early
gains are about reaching the goal *at all*.

## Debugging journey (what went wrong before it worked)

Getting to the result above took unwinding several issues, in order:

### 1. Out-of-memory on the 8 GB GPU
Initial runs at `batch_size=16` **without** mixed precision OOM'd (~5.87 GiB of activations
against a ~7.5 GiB budget). Compounded by **zombie GPU processes** from crashed runs
holding up to ~3.95 GiB. Fixes:
- Added **bf16 autocast** around the forward/loss — roughly halves activation memory.
- Reduced batch size during debugging; cleaned stray processes with `nvidia-smi` + `kill`.

After the real fixes (below), `trm_1m` peaks well under 2 GB, so memory is no longer a constraint.

### 2. Gradient checkpointing silently broke training
To save memory during the OOM firefight, `torch.utils.checkpoint(use_reentrant=False)`
was added inside the bf16 autocast. This **produced wrong gradients**: activations are
recomputed *outside* the autocast context on the backward pass, so the bf16 training loss
fell and `train_acc` rose, but the float32 **eval metrics were frozen** — identical
`success 0.005 / cell_acc 0.364` across epochs. Removed checkpointing entirely; at this
model scale the memory saving wasn't needed.

### 3. EMA was missing buffers
The first EMA implementation only shadowed `requires_grad` parameters, omitting registered
**buffers** (the 2D RoPE cos/sin tables). This made the EMA-evaluated model subtly stale.
Fixed by shadowing the full `state_dict()` (params **and** buffers).

### 4. The real blocker: learning rate too high
Even with the above fixed, the model sat at `cell_acc ≈ 0.36`, `success ≈ 0.005`,
**predicting a single action** (North or South — the two ~26%-frequent classes). This is a
saddle-point trap: predicting the marginal-majority action is locally optimal, and at high
lr the gradient noise across diverse grids drowns out the weak signal needed to escape.

We confirmed the **architecture was fine** with a single-batch overfit test — the model
drives loss 5.0 → 0.20 and accuracy 0.38 → 0.98 on 4 fixed grids, in **both float32 and
bf16**. So the full-data stall was purely an optimization problem.

A learning-rate / batch sweep (500 steps, full data) pinned it down:

| lr | batch | Outcome |
|------|------:|---------|
| **1.5e-3** | 8 | ❌ stuck — loss 4.12, acc 0.45, predicts 1 of 5 actions |
| 3e-4 | 8 | ✅ escaped — loss 1.70, acc 0.80 |
| 1e-3 | 32 | ✅ escaped — loss 3.09, acc 0.61 |
| 3e-4 | 32 | ✅ escaped — loss 0.67, acc **0.94** |

**Dropping lr from 1.5e-3 → 3e-4 was the fix.** A larger batch helps further (cleaner
gradients → faster, more complete escape). Note: raising lr when increasing batch (the
standard linear-scaling heuristic) made things *worse* here — that rule assumes you're
already in a stable regime, which this saddle-trapped model was not.

### 5. Shrinking the model
With a working recipe, width and reasoning depth were cut for speed: `dim 128 → 64`,
`num_heads 8 → 4`, `n 6 → 4`. This dropped params **4.82M → 1.2M** (~4×) and roughly
halved per-step compute, yet the small model still reaches **100% success** — the task
simply doesn't need a large model at this grid size / density.

## Open work

- **Generalization across grid size and obstacle density.** All results are on 26×26 @ 25%.
  Evaluate the trained checkpoint on held-out grids of varying size (e.g. 16×16, 40×40) and
  density (e.g. 0.10, 0.35) to measure out-of-distribution robustness. 2D RoPE is relative,
  so some size extrapolation is plausible but unverified.
- **CNN baseline & `n` ablations** (`configs/ablation_n*.yaml`, `cnn_baseline.yaml`) to
  quantify the value of recursive reasoning depth.
- **Weights & Biases** logging integration (currently CSV + rerun only).
- **Early stopping** — `trm_1m` plateaus by epoch ~6, so `num_epochs` can be small.
