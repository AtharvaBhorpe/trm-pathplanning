"""
Out-of-distribution (OOD) generalization: how well do the learned planners hold up on
grid sizes and obstacle densities they never saw in training? trm_1m / cnn_* are trained
*only* on 26x26 @ 25% (see results.md), so this is the key open question.

Two independent sweeps, each holding the other axis at the training value:
    density : 26x26 grids at densities {0.10, 0.15, 0.25, 0.35, 0.40}
    size    : {16, 20, 26, 32, 40}-side grids at 25% density
(26x26 @ 0.25 is the in-distribution anchor in both — it should reproduce the ~0.99
training-time success and validates the whole pipeline.)

Why this is even possible across *sizes*: the learned weights are per-token and
size-independent. The TRM's only size-dependent state is its 2D-RoPE cos/sin buffers,
which we simply rebuild for the new side length (RoPE is *relative*, so this is genuine
extrapolation, not a hack). CNNs are fully convolutional, so they're size-agnostic for
free — but their fixed receptive field caps how far they can plan. A* is the oracle
anchor (optimal by construction at any size/density).

Test grids are generated on demand (test-only, no train/val) into --data-dir and cached.

Run:
    uv run python -m scripts.eval_ood
    uv run python -m scripts.eval_ood --n 1000 --batch-size 64
    uv run python -m scripts.eval_ood --sizes 16 26 40 --densities 0.10 0.25 0.40
"""
import argparse
import os

import numpy as np
import torch

from dataset import build_dataset as B
from dataset.astar import astar_path_length
from dataset.loader import make_loader
from dataset.solver import optimal_path_length
from pretrain import build_model, evaluate
from utils import rerun_viz

IN_SIZE, IN_DENSITY = 26, 0.25  # the training distribution (in-dist anchor)


def ensure_dataset(data_dir, size, density, n, seed):
    """Generate (and cache) a test-only parquet for one (size, density) condition."""
    path = os.path.join(data_dir, f"grids_{size}x{size}_d{int(round(density * 100))}_test.parquet")
    if not os.path.exists(path):
        print(f"  generating {n} test grids: {size}x{size} @ {density:.2f} -> {path}")
        cols = B.build_split(n, size, density, seed)
        B.write_parquet(path, *cols, size)
    return path


def load_model_at_size(ckpt_path, grid_size, device):
    """Rebuild a checkpoint's model at a NEW grid size and load its size-independent weights.

    The TRM's RoPE cos/sin buffers are size-dependent, so we drop them from the loaded
    state_dict and let the freshly-built model's correctly-sized buffers stand. All learned
    parameters (and CNN BatchNorm stats) are size-independent and must load cleanly.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg["data"]["grid_size"] = grid_size
    model = build_model(cfg, device)

    filtered = {k: v for k, v in ckpt["state_dict"].items() if "rope" not in k}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    learned_missing = [k for k in missing if "rope" not in k]
    assert not learned_missing, f"missing learned params at size {grid_size}: {learned_missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"

    name = ckpt_path.rstrip("/").split("/")[-2]  # checkpoints/trm_1m/best.pt -> trm_1m
    return name, model.eval()


def astar_oracle(data_path, n):
    """Confirm A* stays optimal on each condition (success/optimality, should be ~1.0/1.0)."""
    samples = rerun_viz.load_viz_samples(data_path, n)
    succ, ratios = 0, []
    for s in samples:
        a_len = astar_path_length(s["occ"], s["start"], s["goal"])
        opt = optimal_path_length(s["occ"], s["start"], s["goal"])
        if a_len > 0:
            succ += 1
            ratios.append(a_len / opt)
    return succ / len(samples), (float(np.mean(ratios)) if ratios else 0.0)


def print_matrix(title, axis_label, conditions, methods, table, metric):
    """table[cond][method] = metrics dict. Print one markdown table for `metric`."""
    print(f"\n### {title} — {metric}\n")
    print(f"| {axis_label} | " + " | ".join(methods) + " |")
    print("|" + "---|" * (len(methods) + 1))
    for c in conditions:
        cells = []
        for m in methods:
            v = table[c].get(m, {}).get(metric)
            cells.append("—" if v is None else f"{v:.3f}")
        print(f"| {c} | " + " | ".join(cells) + " |")


def run_condition(data_dir, size, density, n, seed, batch_size, ckpts, device):
    """Eval A* + every checkpoint on one (size, density) condition; return {method: metrics}."""
    path = ensure_dataset(data_dir, size, density, n, seed)
    loader = make_loader(path, batch_size, shuffle=False, num_workers=4)
    grid_side = loader.dataset.size

    out = {}
    a_succ, a_opt = astar_oracle(path, n)
    out["A* (oracle)"] = {"success_rate": a_succ, "optimality_ratio": a_opt, "per_cell_acc": None}
    for ckpt in ckpts:
        name, model = load_model_at_size(ckpt, grid_side, device)
        out[name] = evaluate(model, loader, grid_side, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description="OOD generalization across grid size and density.")
    ap.add_argument("--trm", default="checkpoints/trm_1m/best.pt")
    ap.add_argument("--cnn", nargs="*",
                    default=["checkpoints/cnn_deep/best.pt", "checkpoints/cnn_shallow/best.pt"])
    ap.add_argument("--data-dir", default="data/ood")
    ap.add_argument("--sizes", type=int, nargs="*", default=[16, 20, 26, 32, 40])
    ap.add_argument("--densities", type=float, nargs="*", default=[0.10, 0.15, 0.25, 0.35, 0.40])
    ap.add_argument("--n", type=int, default=2000, help="test grids per condition")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1000, help="base seed for OOD grid generation")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = [args.trm] + list(args.cnn)
    # column order: oracle first, then the learned models in the order given
    methods = ["A* (oracle)"] + [c.rstrip("/").split("/")[-2] for c in ckpts]

    # --- density sweep (size held at the training value) ---
    dens_tbl = {}
    print(f"\n=== density sweep @ {IN_SIZE}x{IN_SIZE} ===")
    for d in args.densities:
        seed = args.seed + IN_SIZE * 100 + int(round(d * 100))
        dens_tbl[f"{d:.2f}"] = run_condition(
            args.data_dir, IN_SIZE, d, args.n, seed, args.batch_size, ckpts, device)

    # --- size sweep (density held at the training value) ---
    size_tbl = {}
    print(f"\n=== size sweep @ density {IN_DENSITY} ===")
    for s in args.sizes:
        seed = args.seed + s * 100 + int(round(IN_DENSITY * 100))
        size_tbl[f"{s}x{s}"] = run_condition(
            args.data_dir, s, IN_DENSITY, args.n, seed, args.batch_size, ckpts, device)

    # --- report ---
    print(f"\n\n## OOD results (n={args.n} grids/condition; in-dist = {IN_SIZE}x{IN_SIZE} @ {IN_DENSITY})")
    dens_conds = [f"{d:.2f}" for d in args.densities]
    size_conds = [f"{s}x{s}" for s in args.sizes]
    for metric in ("success_rate", "optimality_ratio", "per_cell_acc"):
        print_matrix("Density sweep (26x26)", "density", dens_conds, methods, dens_tbl, metric)
    for metric in ("success_rate", "optimality_ratio", "per_cell_acc"):
        print_matrix("Size sweep (@0.25)", "grid", size_conds, methods, size_tbl, metric)


if __name__ == "__main__":
    main()
