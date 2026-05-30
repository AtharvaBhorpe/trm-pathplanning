"""
Accuracy comparison on the test split: classical A* (the oracle) vs the learned models
(CNN shallow, CNN deep, TRM).

All learned models are scored with the SAME evaluation used during training
(`pretrain.evaluate` -> `utils.metrics`), so the TRM numbers here match its training-time
best. A* is optimal by construction, so it anchors the table at success=1.0 / optimality=1.0
(it produces a single path, not a per-cell action field, so per-cell accuracy is N/A).

The interesting comparison is shallow-CNN vs deep-CNN vs TRM: same ~1.2M params, but the
shallow CNN's 9x9 receptive field can't plan across the 26x26 grid, so its success rate
should lag despite decent per-cell accuracy.

Run:
    uv run python -m scripts.compare_baselines
    uv run python -m scripts.compare_baselines --data data/grids_26x26_d25_test.parquet
"""
import argparse

import numpy as np
import torch

from dataset.astar import astar_path_length
from dataset.loader import make_loader
from dataset.solver import optimal_path_length
from pretrain import build_model, evaluate
from utils import rerun_viz


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"], device)
    model.load_state_dict(ckpt["state_dict"])
    params = sum(p.numel() for p in model.parameters())
    name = ckpt_path.rstrip("/").split("/")[-2]
    return name, params, model


def astar_oracle(data_path, n):
    """Empirically confirm A* is optimal on the test grids (should be 1.0 / 1.0)."""
    samples = rerun_viz.load_viz_samples(data_path, n)
    succ, ratios = 0, []
    for s in samples:
        a_len = astar_path_length(s["occ"], s["start"], s["goal"])
        opt = optimal_path_length(s["occ"], s["start"], s["goal"])
        if a_len > 0:
            succ += 1
            ratios.append(a_len / opt)
    return succ / len(samples), float(np.mean(ratios))


def main():
    ap = argparse.ArgumentParser(description="Accuracy comparison on the test split.")
    ap.add_argument("--data", default="data/grids_26x26_d25_test.parquet")
    ap.add_argument("--trm", default="checkpoints/trm_1m/best.pt")
    ap.add_argument("--cnn", nargs="*",
                    default=["checkpoints/cnn_shallow/best.pt", "checkpoints/cnn_deep/best.pt"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--astar-n", type=int, default=1000, help="grids to confirm A* optimality")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_loader(args.data, args.batch_size, shuffle=False, num_workers=4)
    size = loader.dataset.size

    rows = []  # (label, params_str, success, optimality, cellacc_str)

    # A* oracle (optimal by construction; verify empirically)
    a_succ, a_opt = astar_oracle(args.data, args.astar_n)
    rows.append(("A* (oracle)", "—", a_succ, a_opt, "—"))

    # learned models, scored with the training-time evaluator
    for ckpt in [args.trm] + list(args.cnn):
        name, params, model = load_model(ckpt, device)
        m = evaluate(model, loader, size, device)
        rows.append((name, f"{params / 1e6:.2f}M",
                     m["success_rate"], m["optimality_ratio"], f"{m['per_cell_acc']:.3f}"))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n### Accuracy on {args.data} (n={len(loader.dataset)})\n")
    print("| method | params | success | optimality | per-cell acc |")
    print("|---|---|---|---|---|")
    for label, params, succ, opt, cellacc in rows:
        print(f"| {label} | {params} | {succ:.3f} | {opt:.3f} | {cellacc} |")


if __name__ == "__main__":
    main()
