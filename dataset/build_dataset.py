"""
    Generate random occupancy grids, solve each for its optimal action field,
    and write train/val/test splits to parquet.

    Parquet schema (one row per sample):
        grid_flat   : list<int8>    flattened H*W occupancy (0: free, 1: obstacle)
        action_flat : list<int64>   flattened H*W optimal actions (IGNORE_INDEX masked)
        start_idx   : int32         flattened index of start cell
        goal_idx    : int32         flattened index of goal cell
        grid_size   : int16         H (== W; grids are square)
    
    Usage:
        uv run python -m dataset.build_dataset --config configs/base.yaml
        uv run python -m dataset.build_dataset --config configs/base.yaml --visualize 20
"""
import argparse
import sys
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from solver import optimal_action_field, is_reachable

def make_grid(size: int, density: float, rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Generate a solvable grid with random start and goal."""
    while True:
        grid = (rng.random((size, size)) < density).astype(np.int8)
        free_cells = np.argwhere(grid == 0)
        if len(free_cells) < 2:
            continue  # Not enough free cells for start and goal
        start_idx, goal_idx = rng.choice(len(free_cells), size=2, replace=False)
        start = tuple(free_cells[start_idx])
        goal = tuple(free_cells[goal_idx])
        if start != goal and is_reachable(grid, start, goal):
            return grid, start, goal
        
def build_split(n: int, size: int, density: float, seed: int) -> tuple[list[list[int]], list[list[int]], list[int], list[int]]:
    """Build n samples; returns column arrays for a pyarrow table."""
    rng = np.random.default_rng(seed)
    grid_flats, action_flats, start_idxs, goal_idxs = [], [], [], []

    for k in range(n):
        grid, start, goal = make_grid(size, density, rng)
        action_field = optimal_action_field(grid, goal)

        grid_flats.append(grid.reshape(-1).tolist())
        action_flats.append(action_field.reshape(-1).tolist())
        start_idxs.append(start[0] * size + start[1])
        goal_idxs.append(goal[0] * size + goal[1])
        if (k + 1) % 5000 == 0:
            print(f"  Generated {k + 1}/{n} samples")
    return grid_flats, action_flats, start_idxs, goal_idxs

def write_parquet(path: str, grids: list[list[int]], actions: list[list[int]], starts: list[int], goals: list[int], size: int):
    """Write a dataset split to parquet."""
    table = pa.table({
        "grid_flat":    pa.array(grids, type=pa.list_(pa.int8())),
        "actions_flat": pa.array(actions, type=pa.list_(pa.int64())),
        "start_idx":    pa.array(starts, type=pa.int32()),
        "goal_idx":     pa.array(goals, type=pa.int32()),
        "grid_size":    pa.array([size] * len(grids), type=pa.int16())
    })
    pq.write_table(table, path)
    print(f"Wrote {len(grids)} samples to {path}")

def visualize(path, k):
    """Log k samples to rerun for visual ground-truth verification."""
    from utils import rerun_viz
    rerun_viz.preview_dataset(path, k)

def main():
    parser = argparse.ArgumentParser(description="Build a pathfinding dataset of random occupancy grids and optimal action fields.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--visualize", type=int, default=0, help="Number of samples to visualize with rerun (optional)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d = cfg["data"]
    size, density = d["grid_size"], d["obstacle_density"]
    out = d["data_dir"]
    os.makedirs(out, exist_ok=True)

    if args.visualize > 0:
        print(f"Visualizing {args.visualize} samples from {out}_train.parquet")
        visualize(f"{out}_train.parquet", args.visualize)
        return
    
    seed = cfg.get("seed", 0)
    print(f"Generating {size}x{size} grids with {density:.2f} obstacle density (seed={seed})")
    for split, n, s in [("train", d["num_train"], seed),
                        ("val",   d["num_val"],   seed + 1),
                        ("test",  d["num_test"],  seed + 2)]:
        print(f"Building {split} split with {n} samples...")
        cols = build_split(n, size, density, s)
        write_parquet(f"{out}_{split}.parquet", *cols, size)

if __name__ == "__main__":
    main()