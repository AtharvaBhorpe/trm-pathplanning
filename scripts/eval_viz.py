"""
Evaluate a trained checkpoint on the test split and visualize predicted paths in rerun.

For each sample: shows the grid, the ground-truth optimal path (green), the model's
predicted path (purple), and a SUCCESS / FAILED label. Scrub the 'sample' timeline
in the rerun viewer to step through all visualized samples.

Run:
    uv run python -m scripts.eval_viz --config configs/base.yaml
    uv run python -m scripts.eval_viz --config configs/base.yaml --n 50
    uv run python -m scripts.eval_viz --config configs/base.yaml --checkpoint path/to/best.pt
"""
import argparse
import os

import numpy as np
import torch
import yaml

from models import cnn_baseline
from models.trm import TinyRecursiveModel
from utils import rerun_viz


def _build_model(cfg, device):
    """Rebuild the model architecture from the config stored inside the checkpoint."""
    m = cfg["model"]
    if m["arch"] == "trm":
        model = TinyRecursiveModel(
            dim=m["dim"],
            num_heads=m["num_heads"],
            num_layers=m["num_layers"],
            num_classes=m["num_classes"],
            grid_size=cfg["data"]["grid_size"],
            T=m["T"],
            n=m["n"],
            halting=m["halting"],
        )
    else:
        m["grid_size"] = cfg["data"]["grid_size"]
        model = cnn_baseline.build(m)
    return model.to(device)


@torch.no_grad()
def _predict(model, grid_tokens, device):
    """Run the model on one flat token sequence; returns a flat numpy action array."""
    x = torch.tensor(grid_tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)
    return model(x)["final_logits"].argmax(dim=-1).squeeze(0).cpu().numpy()      # (L,)


def main():
    ap = argparse.ArgumentParser(description="Visualize model predictions on the test split.")
    ap.add_argument("--config", required=True, help="YAML config used during training")
    ap.add_argument("--checkpoint", default=None,
                    help="path to best.pt (default: checkpoint_dir/best.pt from the config)")
    ap.add_argument("--n", type=int, default=20, help="number of test samples to visualize")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint — always use the config saved inside it so the architecture matches
    ckpt_path = args.checkpoint or os.path.join(cfg["logging"]["checkpoint_dir"], "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"loaded checkpoint: epoch {ckpt['epoch']}  ({ckpt_path})")

    model = _build_model(ckpt["config"], device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # load_viz_samples reads the parquet directly so we get raw occ grids (no tokens injected)
    ddir = cfg["data"]["data_dir"]
    samples = rerun_viz.load_viz_samples(f"{ddir}_test.parquet", args.n)
    size = samples[0]["occ"].shape[0]

    rerun_viz.init("trm-eval")
    successes = 0
    for i, s in enumerate(samples):
        predicted = _predict(model, s["grid_tokens"], device).reshape(size, size)

        # check success by tracing the predicted path and seeing if it reaches the goal
        pred_pts = rerun_viz.path_from_actions(s["occ"], s["start"], s["goal"], predicted)
        goal_centre = np.array([s["goal"][1] + 0.5, s["goal"][0] + 0.5], dtype=np.float32)
        if len(pred_pts) > 0 and np.allclose(pred_pts[-1], goal_centre):
            successes += 1

        rerun_viz.log_sample("test", s["occ"], s["start"], s["goal"], s["optimal"],
                             predicted_actions=predicted, step=i, timeline="sample")

    print(f"success rate on {len(samples)} test samples: {successes}/{len(samples)} ({successes / len(samples):.1%})")
    print("scrub the 'sample' timeline in rerun to step through the samples")


if __name__ == "__main__":
    main()
