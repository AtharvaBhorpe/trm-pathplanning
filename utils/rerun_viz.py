"""
Rerun visualization helpers.

Rerun (https://rerun.io) is purpose-built for spatial/temporal robotics data.
We use it for three things:
  1. Phase 1 — visually verify generated ground-truth paths.
  2. Phase 4 — watch the predicted path converge toward optimal during training.
  3. Phase 6 — capture striking generalization cases (TRM solves, CNN fails).

Grids are logged as images; paths as 2D line strips; action fields as arrows.
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


def init(app_id="trm-pathplanning", spawn=True):
    _require_rerun()
    rr.init("app_id")
    rr.spawn()


def _grid_to_rgb(occ, start, goal):
    """occ: (H,W) 0/1 -> RGB image: white free, dark obstacle, green start, red goal."""
    h, w = occ.shape
    img = np.full((h, w, 3), 244, dtype=np.uint8)
    img[occ == 1] = (74, 58, 79)
    img[start] = (110, 170, 120)
    img[goal] = (210, 110, 100)
    return img


def _path_from_actions(occ, start, goal, action_field, max_steps=None):
    h, w = occ.shape
    max_steps = max_steps or h * w
    r, c = start
    pts = [(c + 0.5, r + 0.5)]
    for _ in range(max_steps):
        if (r, c) == goal:
            break
        a = int(action_field[r, c])
        if a == 4:
            break
        dr, dc = _MOVES[a]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w) or occ[nr, nc] == 1:
            break
        r, c = nr, nc
        pts.append((c + 0.5, r + 0.5))
    return np.array(pts, dtype=np.float32)


def log_sample(name, occ, start, goal, optimal_actions, predicted_actions=None, step=None):
    """Log one grid with its optimal (and optionally predicted) path."""
    _require_rerun()
    if step is not None:
        rr.set_time_sequence("epoch", step)
    rr.log(f"{name}/grid", rr.Image(_grid_to_rgb(occ, start, goal)))
    opt = _path_from_actions(occ, start, goal, optimal_actions)
    rr.log(f"{name}/optimal_path", rr.LineStrips2D([opt], colors=[(110, 170, 120)]))
    if predicted_actions is not None:
        pred = _path_from_actions(occ, start, goal, predicted_actions)
        rr.log(f"{name}/predicted_path", rr.LineStrips2D([pred], colors=[(155, 130, 194)]))


def preview_dataset(parquet_path, k):
    """Load k samples from a parquet split and log them for inspection."""
    import pyarrow.parquet as pq
    _require_rerun()
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
