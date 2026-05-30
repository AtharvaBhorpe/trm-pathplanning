# Experiment Log

## Headline result

A **~1.2M-parameter** TRM (`configs/trm_1m.yaml`) reaches **100% path-success** with
**perfectly optimal paths** on 26×26 grids at 25% obstacle density, converging by
**epoch 6** (~30 minutes of training on an 8 GB RTX 5060 Mobile).

| Config | Params | Best val success | Optimality | Cell acc | Epochs to converge |
|--------|-------:|-----------------:|-----------:|---------:|-------------------:|
| `trm_1m` (dim 64, n 4) | 1,205,760 | **1.000** | **1.000** | **1.000** | ~6 |
| `base` (dim 128, n 6) | 4,819,968 | — (reference size) | — | — | — |

## Dataset

- **Grid size:** 26×26 (`L = 676` cells)
- **Obstacle density:** 25% (random, regenerated until start→goal is reachable)
- **Splits:** 40k train / 5k val / 10k test, disjoint seeds
- Ground-truth optimal action field per cell from BFS (`dataset/solver.py`)

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

## Baselines (A*, CNN)

To put the TRM result in context we compared it against a classical planner and
parameter-matched CNNs. Reproduce with:

```bash
uv run python -m scripts.compare_baselines     # accuracy on the test split
uv run python -m scripts.benchmark_latency      # latency / throughput
```

### Accuracy on the 10k test split (26×26 @ 25%)

| Method | Params | Success | Optimality | Per-cell acc |
|--------|-------:|--------:|-----------:|-------------:|
| A* (oracle) | — | **1.000** | **1.000** | — |
| `trm_1m` | 1.21M | **0.999** | **1.000** | 0.999 |
| `cnn_deep` (13 conv, RF 27×27) | 1.21M | 0.898 | 1.019 | 0.938 |
| `cnn_shallow` (4 conv, RF 9×9) | 1.20M | 0.128 | 1.004 | 0.712 |

**The receptive field is the story.** All three learned models have ~1.2M params, but
a stack of `L` 3×3 convs only sees a `(1+2L)×(1+2L)` window. The shallow CNN's 9×9
field physically cannot span a 26×26 grid, so it collapses to **0.128 success** despite
a respectable 0.712 per-cell accuracy — a single wrong action breaks the whole path.
Going deep (13 layers, 27×27 field) recovers most of it (**0.898**), but even a
param-matched deep CNN doesn't catch the TRM, whose weight-shared recursion gives it an
effectively global, iterative view. The TRM matches the A* oracle to within 0.001.

### Latency / throughput (RTX 5060 Mobile, GPU bf16, batch sizes 1–512)

Throughput (grids/sec, higher is better). GPU forwards run **bf16 autocast** — the
precision the models train and deploy in (the default of `benchmark_latency.py`); fp32
over-states neural cost, ~2–3× on the recursive TRM. A*/BFS are CPU-only and don't batch:

| Method | B=1 | B=8 | B=64 | B=512 |
|--------|----:|----:|-----:|------:|
| A* (1 path, CPU) | 16,900 | 16,900 | 16,900 | 16,900 |
| BFS (full field, CPU) | 1,480 | 1,480 | 1,480 | 1,480 |
| `cnn_shallow` (GPU bf16) | 3,250 | 9,360 | 12,900 | 11,300 |
| `cnn_deep` (GPU bf16) | 1,400 | 6,260 | 9,320 | 6,960 |
| `trm_1m` (GPU bf16) | 115 | 317 | 244 | 212 |

`--compile` (`torch.compile`, default mode) fuses the recursion's many small kernels for
a further speedup — biggest on the TRM, which is launch-bound by its 30 layer applications:

| Method | B=1 | B=8 | B=64 | B=512 |
|--------|----:|----:|-----:|------:|
| `cnn_shallow` (GPU bf16+compile) | 2,680 | 9,720 | 14,100 | 14,100 |
| `cnn_deep` (GPU bf16+compile) | 2,020 | 6,740 | 9,790 | 8,520 |
| `trm_1m` (GPU bf16+compile) | **348** | **453** | **448** | **433** |

So the TRM climbs from ~79 grids/s (fp32, the earlier table) → ~212 (bf16) → ~433
(bf16+compile) at B=512 — roughly **5×** end to end, purely from running it the way it
actually deploys. (`max-autotune` goes further still but its CUDA-graph pools OOM on this
8 GB laptop GPU at B=512, so the benchmark uses plain `torch.compile`.)

**Even so, on grids this small classical A* wins on speed.** A* answers a single
start→goal query in ~0.06 ms — the 26×26 search space is tiny — so it out-throughputs
every neural model even though it can't batch. The TRM is still the slowest learned model:
its recursion (T×(n+1) layer applications) and O(L²) self-attention make each forward
expensive, which is the deliberate trade — effective depth far beyond its 1.2M params.
The CNNs are much faster per forward (one pass, not 30) and batch well, but accuracy is
the catch (see above).

The takeaway: **at this grid size the learned planner's value is not raw speed — it's
that A* is the oracle and the TRM nearly matches its accuracy at small, fixed cost.** The
latency crossover where neural amortization wins would appear on much larger grids, where
A*'s search space (and BFS's full-field cost) grows with area while a batched forward does
not. That is exactly the untested regime below.

> Reproduce: `benchmark_latency.py` defaults to GPU bf16; add `--compile` for the fused
> numbers, or `--precision fp32` for the earlier (slower, less representative) table.

## Out-of-distribution generalization (size & density)

The checkpoints are trained **only** on 26×26 @ 25%. Here we hold one axis at the training
value and sweep the other, on fresh test grids (2,000 per condition) the models never saw.
A* is the oracle anchor (optimal at any size/density). Reproduce with:

```bash
uv run python -m scripts.eval_ood     # both sweeps; test grids generated on demand
```

This is possible across *sizes* because the learned weights are per-token and
size-independent — the TRM's only size-dependent state is its 2D-RoPE cos/sin tables, which
we rebuild for the new side length. RoPE is **relative**, so this is genuine extrapolation,
not retraining. CNNs are fully convolutional and size-agnostic for free.

### Density sweep — 26×26, success rate (optimality stays ~1.0 throughout)

| density | A* | `trm_1m` | `cnn_deep` | `cnn_shallow` |
|--------:|---:|---------:|-----------:|--------------:|
| 0.10 | 1.000 | **0.996** | 0.807 | 0.122 |
| 0.15 | 1.000 | **1.000** | 0.854 | 0.135 |
| 0.25 *(in-dist)* | 1.000 | **0.999** | 0.893 | 0.142 |
| 0.35 | 1.000 | **0.996** | 0.869 | 0.152 |
| 0.40 | 1.000 | **0.991** | 0.884 | 0.233 |

**Density is a non-event for the TRM.** It holds ≥0.99 success from sparse (0.10) to dense
(0.40) clutter without ever training on those densities — the global recursion doesn't care
how much of the grid is blocked. The deep CNN trails at ~0.81–0.89 but is also stable;
the shallow CNN stays broken (its 9×9 field can't span the grid regardless of density).

### Size sweep — 25% density, success rate

| grid | A* | `trm_1m` | `cnn_deep` | `cnn_shallow` |
|-----:|---:|---------:|-----------:|--------------:|
| 16×16 | 1.000 | 0.991 | **0.997** | 0.341 |
| 20×20 | 1.000 | **0.996** | 0.962 | 0.209 |
| 26×26 *(in-dist)* | 1.000 | **0.999** | 0.893 | 0.142 |
| 32×32 | 1.000 | **0.943** | 0.666 | 0.087 |
| 40×40 | 1.000 | **0.519** | 0.435 | 0.056 |

**Size is the real test, and the TRM's recursion wins it.** Going *smaller* than trained, the
TRM stays near-perfect (0.99 at 16/20). Going *bigger*, it degrades gracefully — still 0.94
at 32×32 (~1.5× the trained side) before falling to 0.52 at 40×40, where inter-cell distances
and path lengths run well beyond anything 2D-RoPE saw in training. The deep CNN falls faster
at every size ≥20 (0.96 → 0.67 → 0.44), and the gap to the TRM **widens with size** — the
recursion's global, iterative view extrapolates where a fixed receptive field cannot. The one
exception is 16×16, where the deep CNN's 27×27 field finally spans the whole grid and edges
ahead (0.997 vs 0.991). The shallow CNN never recovers.

Two cross-cutting notes: **optimality_ratio stays ~1.00–1.02 across every condition** — when a
model reaches the goal it does so near-optimally, so all the OOD degradation is about reaching
the goal *at all*, not taking detours. And **per-cell accuracy degrades much more slowly than
success** (TRM 0.94 cell-acc at 40×40 despite 0.52 success) — a reminder that a single wrong
action can break an otherwise-correct path, which is why success rate is the honest metric.

**Bottom line:** the TRM generalizes across obstacle density essentially for free, and
extrapolates across grid size markedly better than a parameter-matched CNN, holding up to
~1.5× the trained side before degrading. This is the predicted payoff of weight-shared
recursion + relative position over a fixed-receptive-field convolution.

### Test-time recursion: think longer on bigger grids

The 40×40 drop is partly a *planning-depth* problem, and the TRM has a lever the CNN
structurally lacks: because its layers are weight-shared, it can run **more reasoning
recursions at inference than it trained with** — no retraining, same checkpoint. Sweeping
the inference recursion count `n` on the trained `trm_1m` (n=4) checkpoint at 40×40 @ 0.25:

| `n` (inference) | success | optimality | per-cell acc |
|----------------:|--------:|-----------:|-------------:|
| 4 *(as trained)* | 0.519 | 1.013 | 0.940 |
| 6 | 0.656 | 1.011 | 0.946 |
| 8 | **0.749** | 1.013 | 0.944 |
| 12 | 0.748 | 1.013 | 0.942 |
| 16 | 0.747 | 1.014 | 0.940 |

More recursion lifts 40×40 success **0.52 → 0.75**, then plateaus at n≈8. This isolates two
distinct OOD failure modes:

- **Planning depth** (fixable here): the recursion is BFS-like — each round propagates the
  "which way to the goal" signal one more hop outward from the goal. Bigger grids have longer
  paths, so the far cells need more rounds than n=4 supplies. Extra rounds reach them. Note
  `per_cell_acc` barely moves (0.940 → 0.944): most cells were already right; the gains come
  from finishing a *few* distant cells that were each breaking a whole path — which is why
  **success** jumps while cell-accuracy doesn't.
- **Position distribution** (the residual ~0.25 gap): past n≈8 more rounds don't help, because
  2D-RoPE never saw 40-cell position separations in training, so each per-step update is
  slightly miscalibrated. More iterations of a slightly-wrong update don't converge — this is a
  *learned-position* limit, not a depth limit.

We tested whether **RoPE position interpolation** (the long-context-LLM trick: rescale the
40×40 coordinates back into the trained 0–26 range) could close that residual gap at inference,
without retraining. It doesn't — it *hurts*, monotonically:

| `n` | pos-scale | success | per-cell acc |
|----:|----------:|--------:|-------------:|
| 8 | 1.00 *(no PI)* | **0.749** | 0.944 |
| 8 | 0.85 | 0.711 | 0.954 |
| 8 | 0.75 | 0.441 | 0.872 |
| 8 | 0.65 *(full PI)* | 0.164 | 0.723 |

Interpolation helps LLMs because they need *long-range* relative position and tolerate squashed
*local* resolution. Grid planning is the opposite: the decision that matters ("which of my 4
neighbours steps toward the goal?") lives in the *finest* position differences, and compressing
positions crushes exactly that. The tell: at scale 0.85 per-cell acc *rises* while success
*falls* — PI tidies the global field but corrupts a few critical local steps, and one wrong step
breaks a whole path. So the residual gap genuinely requires *training* on longer separations
(deliberately out of scope here to keep the CNN comparison fair); **~0.75 via test-time
recursion is the honest no-retrain ceiling at 40×40.**

Test-time recursion is reported as an *inference* knob only — training is left identical across
all models (`trm_1m`, both CNNs), so the head-to-head OOD comparison above stays fair. We
deliberately avoid mixed-size *training* for the same reason: it would advantage the TRM by
changing its training distribution while the CNNs' stayed fixed.

## Open work

- **Recover large-grid success** (the new frontier, now that OOD is measured above). The TRM
  halves to 0.52 at 40×40; the natural next step is to test whether it's a *capacity* limit or
  a *training-distribution* limit — e.g. train on mixed grid sizes (16–40) and re-run the size
  sweep, or increase `n`/`T` at the large sizes (more recursion = more effective planning
  depth, which is exactly what longer paths need). Would also surface the latency crossover
  (A* search-space growth vs flat batched forward) on the larger grids.
- **`n` ablations** (`configs/ablation_n*.yaml`) to quantify the value of recursive reasoning
  depth. (The CNN baseline and OOD comparisons are now done — see *Baselines* and
  *Out-of-distribution generalization* above.)
- **Weights & Biases** logging integration (currently CSV + rerun + offline matplotlib).
- **Early stopping** — `trm_1m` plateaus by epoch ~6, so `num_epochs` can be small.
