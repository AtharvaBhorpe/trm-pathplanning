"""
Plot training/validation metrics from the CSV logs as static matplotlib figures.

This is the local-only, no-server complement to the live rerun graphs: it reads the
`train_metrics.csv` / `val_metrics.csv` a run writes to its `log_dir` and renders the
whole history at once (no scrolling / follow-mode), saved as a PNG you can drop into
`results.md`. Pass several runs to overlay them and compare.

Note: learning rate is only streamed to rerun, not the CSVs, so it is not plotted here.

Run:
    uv run python -m scripts.plot_metrics --config configs/trm_1m.yaml
    uv run python -m scripts.plot_metrics --log-dir logs/trm_1m
    uv run python -m scripts.plot_metrics --log-dir logs/trm_1m --log-dir logs/base   # compare
    uv run python -m scripts.plot_metrics --config configs/trm_1m.yaml --out figs/trm_1m.png
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")  # headless: render straight to a file, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

# (csv column, axis title). Train rows are per-step, val rows are per-epoch.
TRAIN_PANELS = [("loss", "train loss"), ("train_acc", "train accuracy")]
VAL_PANELS = [
    ("val_success_rate", "val success rate"),
    ("val_optimality_ratio", "val optimality ratio"),
    ("val_per_cell_acc", "val per-cell accuracy"),
]


def read_csv(path):
    """Read a metrics CSV into {column: [floats]}; returns None if the file is missing."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k, v in row.items():
                cols[k].append(float(v) if v not in ("", None) else float("nan"))
    return cols


def load_run(log_dir):
    """Load a run's train/val metrics from its log_dir. Returns a dict (possibly partial)."""
    return {
        "name": os.path.basename(os.path.normpath(log_dir)),
        "train": read_csv(os.path.join(log_dir, "train_metrics.csv")),
        "val": read_csv(os.path.join(log_dir, "val_metrics.csv")),
    }


def main():
    ap = argparse.ArgumentParser(description="Plot CSV metrics as static figures.")
    ap.add_argument("--config", default=None, help="YAML config (derives log_dir from logging.log_dir)")
    ap.add_argument("--log-dir", action="append", default=[], dest="log_dirs",
                    help="run log dir; repeat to overlay/compare multiple runs")
    ap.add_argument("--out", default=None, help="output PNG path (default: <first log_dir>/metrics.png)")
    args = ap.parse_args()

    log_dirs = list(args.log_dirs)
    if args.config:
        cfg = yaml.safe_load(open(args.config))
        log_dirs.insert(0, cfg["logging"]["log_dir"])
    if not log_dirs:
        ap.error("provide --config and/or at least one --log-dir")

    runs = [load_run(d) for d in log_dirs]

    # 2x3 grid: top row = train panels, bottom row = the three val panels
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    def plot_panel(ax, col, title, split, x_from_epoch):
        for run in runs:
            data = run[split]
            if not data or col not in data:
                continue
            y = data[col]
            x = data["epoch"] if x_from_epoch else range(len(y))
            ax.plot(x, y, label=run["name"], linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("epoch" if x_from_epoch else "logged training step")
        ax.grid(True, alpha=0.3)
        if len(runs) > 1:
            ax.legend(fontsize=8)

    # train panels span the per-step axis (step resets each epoch, so use row index)
    plot_panel(axes[0, 0], *TRAIN_PANELS[0], "train", x_from_epoch=False)
    plot_panel(axes[0, 1], *TRAIN_PANELS[1], "train", x_from_epoch=False)
    axes[0, 2].axis("off")  # spare cell — keeps train/val on separate rows
    for ax, (col, title) in zip(axes[1], VAL_PANELS):
        plot_panel(ax, col, title, "val", x_from_epoch=True)

    fig.suptitle("Training metrics" + (" (comparison)" if len(runs) > 1 else f" — {runs[0]['name']}"))
    fig.tight_layout()

    out = args.out or os.path.join(log_dirs[0], "metrics.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")

    # quick textual summary of best/final val numbers per run
    for run in runs:
        val = run["val"]
        if val and val.get("val_success_rate"):
            best = max(val["val_success_rate"])
            final = val["val_success_rate"][-1]
            print(f"  {run['name']}: best val success {best:.3f}, final {final:.3f} "
                  f"over {len(val['epoch'])} epochs")


if __name__ == "__main__":
    main()
