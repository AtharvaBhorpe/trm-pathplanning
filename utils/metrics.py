"""
Evaluation metrics for grid path planning.

The headline metric is *path success rate*: roll out the predicted policy from
the start and check whether it reaches the goal without hitting an obstacle or
looping. Per-cell accuracy is a weaker proxy (a single wrong action can break a
path), so we report both.
"""
import numpy as np
import torch

from dataset.solver import optimal_path_length
from dataset.loader import OBSTACLE, GOAL_TOKEN

_MOVES = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # N,S,E,W ; index 4 (stay) handled separately


@torch.no_grad()
def per_cell_accuracy(pred, target, ignore_index=-100):
    """Fraction of non-masked cells with the correct action."""
    mask = target != ignore_index
    if mask.sum() == 0:
        return 0.0
    return (pred[mask] == target[mask]).float().mean().item()


def rollout(grid2d, start, goal, action_field, max_steps=None):
    """
    Follow the predicted policy from start. Returns (reached: bool, steps: int).
    grid2d: (H,W) 0/1 occupancy. action_field: (H,W) predicted action ids.
    """
    h, w = grid2d.shape
    max_steps = max_steps or (h * w)
    r, c = start
    for step in range(max_steps):
        if (r, c) == goal:
            return True, step
        a = int(action_field[r, c])
        if a == 4:                       # stay but not at goal -> stuck
            return False, step
        dr, dc = _MOVES[a]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w) or grid2d[nr, nc] == OBSTACLE:
            return False, step           # walked into a wall / off-grid
        r, c = nr, nc
    return False, max_steps


@torch.no_grad()
def evaluate_batch(grids, preds, starts, goals, size):
    """
    grids:  (B, L) input tokens (start/goal markers present)
    preds:  (B, L) predicted action ids
    starts/goals: (B,) flattened indices

    Returns raw per-batch counts so the caller can aggregate over the whole
    dataset (averaging per-batch rates is biased when batches differ in size
    or when a batch has zero successes):
        n          : number of samples in the batch
        successes  : number that reached the goal
        opt_ratios : list of steps/optimal for each successful sample
    """
    b = grids.shape[0]
    successes, opt_ratios = 0, []
    grids_np = grids.cpu().numpy().reshape(b, size, size)
    preds_np = preds.cpu().numpy().reshape(b, size, size)

    for i in range(b):
        occ = (grids_np[i] == OBSTACLE).astype(np.int8)   # back to pure occupancy
        s = (int(starts[i]) // size, int(starts[i]) % size)
        g = (int(goals[i]) // size, int(goals[i]) % size)
        reached, steps = rollout(occ, s, g, preds_np[i])
        if reached:
            successes += 1
            opt = optimal_path_length(occ, s, g)
            if opt > 0:
                opt_ratios.append(steps / opt)
    return {"n": b, "successes": successes, "opt_ratios": opt_ratios}
