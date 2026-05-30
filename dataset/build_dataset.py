"""
Build the dataset: make random grids, solve each for its optimal action field,
and save train/val/test splits as parquet files.

Each row stores one sample:
    grid_flat    : list<int8>    flattened H*W occupancy (0 = free, 1 = obstacle)
    actions_flat : list<int64>   flattened H*W optimal actions (IGNORE_INDEX masked)
    start_idx    : int32         flattened index of the start cell
    goal_idx     : int32         flattened index of the goal cell
    grid_size    : int16         H (grids are square, so H == W)

Usage:
    uv run python -m dataset.build_dataset --config configs/base.yaml
    uv run python -m dataset.build_dataset --config configs/base.yaml --visualize 20
"""
import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dataset.solver import is_reachable, optimal_action_field

# allow running this file directly (not just as a module) from the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def make_grid(size, density, rng):
    """
    Make one random grid with a reachable start and goal.

    Keeps retrying until it draws a grid where the goal is reachable from the
    start (random obstacles can otherwise wall them off).
    Returns (grid, start, goal).
    """
    while True:
        grid = (rng.random((size, size)) < density).astype(np.int8)
        free_cells = np.argwhere(grid == 0)
        if len(free_cells) < 2:
            continue  # need at least two free cells for start and goal
        i, j = rng.choice(len(free_cells), size=2, replace=False)
        start, goal = tuple(free_cells[i]), tuple(free_cells[j])
        if start != goal and is_reachable(grid, start, goal):
            return grid, start, goal


def build_split(n, size, density, seed):
    """Generate n samples; returns four column lists for a parquet table."""
    rng = np.random.default_rng(seed)
    grid_flats, action_flats, start_idxs, goal_idxs = [], [], [], []

    for k in range(n):
        grid, start, goal = make_grid(size, density, rng)
        actions = optimal_action_field(grid, goal)

        grid_flats.append(grid.reshape(-1).tolist())
        action_flats.append(actions.reshape(-1).tolist())
        start_idxs.append(start[0] * size + start[1])  # (row, col) -> flat index
        goal_idxs.append(goal[0] * size + goal[1])
        if (k + 1) % 5000 == 0:
            print(f"  generated {k + 1}/{n} samples")
    return grid_flats, action_flats, start_idxs, goal_idxs


def write_parquet(path, grids, actions, starts, goals, size):
    """Write one split to a parquet file."""
    table = pa.table({
        "grid_flat": pa.array(grids, type=pa.list_(pa.int8())),
        "actions_flat": pa.array(actions, type=pa.list_(pa.int64())),
        "start_idx": pa.array(starts, type=pa.int32()),
        "goal_idx": pa.array(goals, type=pa.int32()),
        "grid_size": pa.array([size] * len(grids), type=pa.int16()),
    })
    pq.write_table(table, path)
    print(f"wrote {len(grids)} samples to {path}")


def visualize(path, k):
    """Log k samples to rerun for a visual ground-truth check."""
    from utils import rerun_viz
    rerun_viz.preview_dataset(path, k)


def main():
    parser = argparse.ArgumentParser(description="Build a pathfinding dataset.")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--visualize", type=int, default=0,
                        help="visualize this many train samples with rerun, then exit")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d = cfg["data"]
    size, density, out = d["grid_size"], d["obstacle_density"], d["data_dir"]
    os.makedirs(out, exist_ok=True)

    if args.visualize > 0:
        print(f"visualizing {args.visualize} samples from {out}_train.parquet")
        visualize(f"{out}_train.parquet", args.visualize)
        return

    # different seed per split so train/val/test don't overlap
    seed = cfg.get("seed", 0)
    print(f"generating {size}x{size} grids at {density:.2f} obstacle density (seed={seed})")
    for split, n, split_seed in [
        ("train", d["num_train"], seed),
        ("val", d["num_val"], seed + 1),
        ("test", d["num_test"], seed + 2),
    ]:
        print(f"building {split} split ({n} samples)...")
        cols = build_split(n, size, density, split_seed)
        write_parquet(f"{out}_{split}.parquet", *cols, size)


if __name__ == "__main__":
    main()
