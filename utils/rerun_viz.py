"""
Rerun visualization helpers.

Rerun (https://rerun.io) is purpose-built for spatial/temporal data.
We use it for three things:

  Phase 1 — verify generated ground-truth paths:
      build_dataset.py --visualize 20

  Phase 4 — watch training live and review it afterwards:
      pretrain.py streams scalar metric graphs (loss / acc / lr / val metrics) plus a
      fixed set of predicted-vs-optimal val paths each epoch when viz_samples > 0, and
      also saves everything to logs/<run>/train.rrd so the graphs survive the run
      (reopen with: rerun logs/<run>/train.rrd)

  Phase 6 — inspect model predictions on the test split:
      uv run python -m scripts.eval_viz --config configs/base.yaml

Grids are logged as images; paths as 2D line strips; success/failure as text logs.
"""

import numpy as np

try:
    import rerun as rr
except ImportError:
    rr = None

_MOVES = [(-1, 0), (1, 0), (0, 1), (0, -1)]


def _require_rerun():
    if rr is None:
        raise ImportError("rerun-sdk not installed. Run: uv add rerun-sdk")


def init(app_id="trm-pathplanning", spawn=True, save_path=None):
    """
    Start a rerun recording.

    spawn      : open a live viewer window to watch the run in real time.
    save_path  : if given, also persist everything to a `.rrd` file so the graphs
                 survive after the run ends. Reopen later with: rerun <save_path>.

    With both set, we stream live AND save to disk (rr.spawn(connect=False) launches
    the viewer process without claiming the sink, then set_sinks tees to both).
    """
    _require_rerun()
    rr.init(app_id)
    if save_path:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        sinks = [rr.FileSink(save_path)]
        if spawn:
            rr.spawn(connect=False)  # launch the viewer without grabbing the sink
            sinks.append(rr.GrpcSink())  # ...then also stream to it live
        rr.set_sinks(*sinks)
    elif spawn:
        rr.spawn()


def log_scalars(values: dict, step: int, timeline: str = "step"):
    """
    Stream training/eval metrics to rerun as scalar time-series (Phase 4).

    values   : {entity_path: number}, e.g. {"train/loss": 0.24, "train/lr": 3e-4}.
               Each key becomes its own line plot in the rerun viewer.
    step      : x-axis position on `timeline` (global step for train, epoch for val).
    timeline  : which timeline to plot against ("step" or "epoch").
    """
    _require_rerun()
    rr.set_time(timeline, sequence=step)
    for name, value in values.items():
        rr.log(name, rr.Scalars(float(value)))


def _grid_to_rgb(occ, start, goal):
    """occ: (H,W) 0/1 -> RGB image: white free, dark obstacle, green start, red goal."""
    h, w = occ.shape
    img = np.full((h, w, 3), 244, dtype=np.uint8)
    img[occ == 1] = (74, 58, 79)
    img[start] = (110, 170, 120)
    img[goal] = (210, 110, 100)
    return img


def path_from_actions(occ, start, goal, action_field, max_steps=None):
    """
    Trace a path from start by following action_field until the goal or a dead end.

    Returns an (N, 2) float32 array of (x, y) centre-points in image space
    (x = col + 0.5, y = row + 0.5) — the format rr.LineStrips2D expects.
    """
    h, w = occ.shape
    max_steps = max_steps or h * w
    r, c = start
    pts = [(c + 0.5, r + 0.5)]
    for _ in range(max_steps):
        if (r, c) == goal:
            break
        a = int(action_field[r, c])
        if a == 4:  # STAY — reached goal or stuck
            break
        dr, dc = _MOVES[a]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w) or occ[nr, nc] == 1:
            break
        r, c = nr, nc
        pts.append((c + 0.5, r + 0.5))
    return np.array(pts, dtype=np.float32)


def log_sample(
    name,
    occ,
    start,
    goal,
    optimal_actions,
    predicted_actions=None,
    step=None,
    timeline="sample",
):
    """
    Log one grid with its optimal path (green) and optionally the predicted path (purple).

    step / timeline: if step is given, set_time_sequence(timeline, step) so you can
    scrub through samples (timeline="sample") or training epochs (timeline="epoch").
    """
    _require_rerun()
    if step is not None:
        rr.set_time(timeline, sequence=step)

    rr.log(f"{name}/grid", rr.Image(_grid_to_rgb(occ, start, goal)))

    opt_pts = path_from_actions(occ, start, goal, optimal_actions)
    rr.log(f"{name}/optimal_path", rr.LineStrips2D([opt_pts], colors=[(110, 170, 120)]))

    if predicted_actions is not None:
        pred_pts = path_from_actions(occ, start, goal, predicted_actions)
        rr.log(
            f"{name}/predicted_path",
            rr.LineStrips2D([pred_pts], colors=[(155, 130, 194)]),
        )

        # did the predicted path reach the goal?
        goal_centre = np.array([goal[1] + 0.5, goal[0] + 0.5], dtype=np.float32)
        success = len(pred_pts) > 0 and np.allclose(pred_pts[-1], goal_centre)
        rr.log(
            f"{name}/result",
            rr.TextLog(
                "SUCCESS" if success else "FAILED",
                level=rr.TextLogLevel.INFO if success else rr.TextLogLevel.WARN,
            ),
        )


def load_viz_samples(parquet_path, k):
    """
    Load k samples from a parquet file for training-time visualization.

    Returns a list of dicts, each with:
        occ          : (H, W) int8 occupancy grid (raw, no tokens injected)
        optimal      : (H, W) int64 ground-truth action field
        start, goal  : (row, col) tuples
        grid_tokens  : (H*W,) int64 with START_TOKEN=2 / GOAL_TOKEN=3 injected
                       — ready to be passed directly to the model
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    size = int(table["grid_size"][0].as_py())
    n = min(k, table.num_rows)
    samples = []
    for i in range(n):
        occ = np.array(table["grid_flat"][i].as_py(), dtype=np.int8).reshape(size, size)
        optimal = np.array(table["actions_flat"][i].as_py(), dtype=np.int64).reshape(
            size, size
        )
        si = int(table["start_idx"][i].as_py())
        gi = int(table["goal_idx"][i].as_py())
        start = (si // size, si % size)
        goal = (gi // size, gi % size)
        grid_tokens = np.array(table["grid_flat"][i].as_py(), dtype=np.int64)
        grid_tokens[si] = 2  # START_TOKEN
        grid_tokens[gi] = 3  # GOAL_TOKEN
        samples.append(
            {
                "occ": occ,
                "optimal": optimal,
                "start": start,
                "goal": goal,
                "grid_tokens": grid_tokens,
            }
        )
    return samples


def preview_dataset(parquet_path, k):
    """Load k samples from a parquet split and log them for inspection (Phase 1)."""
    _require_rerun()
    import pyarrow.parquet as pq

    init()
    t = pq.read_table(parquet_path)
    size = int(t["grid_size"][0].as_py())
    for i in range(min(k, t.num_rows)):
        occ = np.array(t["grid_flat"][i].as_py(), dtype=np.int8).reshape(size, size)
        act = np.array(t["actions_flat"][i].as_py(), dtype=np.int64).reshape(size, size)
        si, gi = int(t["start_idx"][i].as_py()), int(t["goal_idx"][i].as_py())
        start, goal = (si // size, si % size), (gi // size, gi % size)
        rr.set_time("sample", sequence=i)
        log_sample("preview", occ, start, goal, act)
    print(f"Logged {min(k, t.num_rows)} samples to rerun. Scrub the 'sample' timeline.")
