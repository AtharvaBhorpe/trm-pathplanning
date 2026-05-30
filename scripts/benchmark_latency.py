"""
Latency / throughput benchmark: classical planners (A*, BFS) vs the learned models
(CNN, TRM) on the test grids.

The headline contrast: classical search runs one grid at a time, so its per-grid cost
is flat no matter how many grids you have — there is no batch to amortize. The neural
models run a whole batch in one forward pass, so their per-grid cost *falls* as the
batch grows and the GPU fills up. This is the core "why a learned planner" argument:
not accuracy (A* is optimal), but throughput under load.

GPU forwards default to bf16 autocast (the precision the models train/deploy in) — fp32
over-states neural latency, ~2-3x on the recursive TRM. `--compile` additionally fuses
the recursion's many small ops (~2x further on the TRM). CPU always runs fp32.

Run:
    uv run python -m scripts.benchmark_latency                       # bf16 GPU forwards
    uv run python -m scripts.benchmark_latency --compile             # + torch.compile (GPU)
    uv run python -m scripts.benchmark_latency --precision fp32      # old fp32 numbers
    uv run python -m scripts.benchmark_latency --trm checkpoints/trm_1m/best.pt \
        --cnn checkpoints/cnn_shallow/best.pt checkpoints/cnn_deep/best.pt
"""
import argparse
import statistics
import time
from contextlib import nullcontext

import numpy as np
import torch

from dataset.astar import astar
from dataset.solver import optimal_action_field
from models import cnn_baseline
from models.trm import TinyRecursiveModel
from utils import rerun_viz

BATCH_SIZES = [1, 8, 64, 512]


def build_model(cfg, device):
    """Rebuild a model from the config stored inside its checkpoint (mirrors eval_viz)."""
    m = cfg["model"]
    if m["arch"] == "trm":
        model = TinyRecursiveModel(
            dim=m["dim"], num_heads=m["num_heads"], num_layers=m["num_layers"],
            num_classes=m["num_classes"], grid_size=cfg["data"]["grid_size"],
            T=m["T"], n=m["n"], halting=m["halting"],
        )
    else:
        m["grid_size"] = cfg["data"]["grid_size"]
        model = cnn_baseline.build(m)
    return model.to(device).eval()


def load_model(ckpt_path, device):
    """Load a checkpoint and return (label, model). Label = arch + param count."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"], device)
    model.load_state_dict(ckpt["state_dict"])
    params = sum(p.numel() for p in model.parameters())
    arch = ckpt["config"]["model"]["arch"].upper()
    name = ckpt_path.rstrip("/").split("/")[-2]  # e.g. checkpoints/cnn_deep/best.pt -> cnn_deep
    return f"{name} ({arch}, {params / 1e6:.2f}M)", model


@torch.no_grad()
def time_neural(model, token_pool, device, batch_size, repeats, warmup, autocast=False):
    """Median per-grid latency (ms) and throughput (grids/s) for one batched forward.

    `autocast` runs the forward under bf16 autocast — the precision the models are
    actually trained/deployed in. fp32 over-states neural latency (~2-3x on the TRM).
    Warmup also absorbs torch.compile's one-time per-shape compilation.
    """
    x = token_pool[:batch_size].to(device)
    ctx = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if autocast else nullcontext
    for _ in range(warmup):
        with ctx():
            model(x)["final_logits"].argmax(dim=-1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with ctx():
            model(x)["final_logits"].argmax(dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return med / batch_size * 1000.0, batch_size / med


def time_classical(fn, samples):
    """Per-grid latency (ms) and throughput (grids/s) — batch-independent (no batching)."""
    t0 = time.perf_counter()
    for s in samples:
        fn(s)
    dt = time.perf_counter() - t0
    n = len(samples)
    return dt / n * 1000.0, n / dt


def print_table(title, rows, unit):
    """rows: list of (label, {batch: value}). Print a markdown table over BATCH_SIZES."""
    print(f"\n### {title} ({unit})\n")
    head = "| method | " + " | ".join(f"B={b}" for b in BATCH_SIZES) + " |"
    print(head)
    print("|" + "---|" * (len(BATCH_SIZES) + 1))
    for label, vals in rows:
        cells = " | ".join(f"{vals[b]:.3g}" if b in vals else "—" for b in BATCH_SIZES)
        print(f"| {label} | {cells} |")


def main():
    ap = argparse.ArgumentParser(description="Latency/throughput benchmark.")
    ap.add_argument("--data", default="data/grids_26x26_d25_test.parquet")
    ap.add_argument("--trm", default="checkpoints/trm_1m/best.pt")
    ap.add_argument("--cnn", nargs="*",
                    default=["checkpoints/cnn_shallow/best.pt", "checkpoints/cnn_deep/best.pt"])
    ap.add_argument("--n", type=int, default=512, help="grid pool size (>= max batch)")
    ap.add_argument("--repeats", type=int, default=20, help="timed forwards per measurement")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--classical-n", type=int, default=300, help="grids to time A*/BFS over")
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                    help="GPU forward precision. bf16 (default) matches training/deployment; "
                         "fp32 over-states neural latency. CPU always runs fp32.")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the neural models (GPU only). Fuses the recursion's "
                         "many small ops — ~2x further on the TRM. Adds one-time warmup.")
    args = ap.parse_args()

    n = max(args.n, max(BATCH_SIZES))
    samples = rerun_viz.load_viz_samples(args.data, n)
    token_pool = torch.tensor(
        np.stack([s["grid_tokens"] for s in samples]), dtype=torch.long
    )  # (n, L)
    print(f"loaded {len(samples)} grids from {args.data}")

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    lat_rows, thr_rows = [], []

    # --- classical (CPU only, no batching: flat across batch sizes) ---
    classical_set = samples[: args.classical_n]
    for label, fn in [
        ("A* (1 path, CPU)", lambda s: astar(s["occ"], s["start"], s["goal"])),
        ("BFS (full field, CPU)", lambda s: optimal_action_field(s["occ"], s["goal"])),
    ]:
        ms, thr = time_classical(fn, classical_set)
        lat_rows.append((label, {b: ms for b in BATCH_SIZES}))
        thr_rows.append((label, {b: thr for b in BATCH_SIZES}))
        print(f"  timed {label}: {ms:.3g} ms/grid")

    # --- neural (CPU + GPU, batched) ---
    # bf16 autocast is applied on GPU only (CPU bf16 isn't representative); compile is GPU-only.
    if args.precision == "bf16" and any(d.type == "cuda" for d in devices):
        torch.set_float32_matmul_precision("high")
    model_specs = [args.trm] + list(args.cnn)
    for device in devices:
        autocast = args.precision == "bf16" and device.type == "cuda"
        warmup = max(args.warmup, 8) if args.compile and device.type == "cuda" else args.warmup
        for ckpt in model_specs:
            label, model = load_model(ckpt, device)
            tag = []
            if autocast:
                tag.append("bf16")
            if args.compile and device.type == "cuda":
                model = torch.compile(model)
                tag.append("compiled")
            suffix = (", " + "+".join(tag)) if tag else ""
            row_label = f"{label} [{device.type}{suffix}]"
            lat, thr = {}, {}
            for b in BATCH_SIZES:
                ms, gps = time_neural(model, token_pool, device, b,
                                      args.repeats, warmup, autocast=autocast)
                lat[b], thr[b] = ms, gps
            lat_rows.append((row_label, lat))
            thr_rows.append((row_label, thr))
            print(f"  timed {row_label}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print_table("Per-grid latency", lat_rows, "ms/grid, lower is better")
    print_table("Throughput", thr_rows, "grids/sec, higher is better")
    print("\nNote: A*/BFS don't batch — their per-grid cost is flat across batch sizes.")


if __name__ == "__main__":
    main()
