"""
Train a TRM (or the CNN baseline) to predict the optimal action at every cell.

Run:  uv run python pretrain.py --config configs/base.yaml

Each epoch: train over the data, evaluate on the validation split, and save the
best checkpoint (by path success rate) to the config's checkpoint_dir.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.loader import make_loader
from models import cnn_baseline
from models.trm import TinyRecursiveModel
from utils import metrics as M
from utils import rerun_viz
from utils.ema import EMA
from utils.logging import CSVLogger

IGNORE_INDEX = -100  # cells the loss skips (obstacles / unreachable)


def set_seed(seed):
    """Make a run reproducible across python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg, device):
    """Create the model named in the config ('trm' or 'cnn') and move it to device."""
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


def compute_loss(output, targets, halt_weight=1.0, deep_supervision=True):
    """
    Loss = per-cell cross-entropy, summed over the model's T supervision steps,
    plus an optional halting term.

    output: dict from the model; 'logits' is a list of T (B, L, C) tensors.
    targets: (B, L) correct action ids; masked cells hold IGNORE_INDEX.
    deep_supervision: if False, only the final step is supervised.

    Returns {'total_loss': tensor, 'final_acc': float (for logging)}.
    """
    steps = range(len(output["logits"])) if deep_supervision else [-1]
    total = 0.0
    for t in steps:
        logits = output["logits"][t]
        # main objective: the right action at every non-masked cell
        total = total + F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        # optional halt head: predict the fraction of cells already correct
        if output["halt_probs"]:
            halt_prob = output["halt_probs"][t].squeeze(-1)  # (B,)
            with torch.no_grad():
                correct = (logits.argmax(-1) == targets) & (targets != IGNORE_INDEX)
                done_fraction = correct.float().mean(dim=1)  # (B,)
            total = total + halt_weight * F.binary_cross_entropy_with_logits(
                halt_prob.float(), done_fraction
            )

    final_acc = M.per_cell_accuracy(
        output["logits"][-1].argmax(-1), targets, IGNORE_INDEX
    )
    return {"total_loss": total, "final_acc": final_acc}


@torch.no_grad()
def evaluate(model, loader, size, device):
    """
    Run the model over a data split and aggregate metrics over the WHOLE split
    (not per-batch averages, which are biased when batches differ in size):
        success_rate     : fraction of grids whose predicted path reaches the goal
        optimality_ratio : path length / shortest length, over successful grids
        per_cell_acc     : fraction of cells with the correct action
    """
    model.eval()
    total_n, total_success = 0, 0
    opt_ratios = []
    cell_correct, cell_total = 0, 0
    for grids, targets in loader:
        grids = grids.to(device)
        preds = model(grids)["final_logits"].argmax(dim=-1).cpu()  # (B, L)
        # recover start/goal from the marker tokens in the input
        starts = (grids == 2).float().argmax(dim=1).cpu().numpy()
        goals = (grids == 3).float().argmax(dim=1).cpu().numpy()

        b = M.evaluate_batch(grids, preds, starts, goals, size)
        total_n += b["n"]
        total_success += b["successes"]
        opt_ratios.extend(b["opt_ratios"])

        mask = targets != IGNORE_INDEX
        cell_correct += (preds[mask] == targets[mask]).sum().item()
        cell_total += int(mask.sum().item())

    return {
        "success_rate": total_success / max(total_n, 1),
        "optimality_ratio": float(np.mean(opt_ratios)) if opt_ratios else 0.0,
        "per_cell_acc": cell_correct / max(cell_total, 1),
    }


@torch.no_grad()
def _log_viz_epoch(model, viz_samples, epoch, device, size):
    """Log predicted vs optimal paths for a fixed set of val samples (Phase 4)."""
    model.eval()
    for j, s in enumerate(viz_samples):
        x = torch.tensor(s["grid_tokens"], dtype=torch.long, device=device).unsqueeze(0)
        pred = (
            model(x)["final_logits"]
            .argmax(dim=-1)
            .squeeze(0)
            .cpu()
            .numpy()
            .reshape(size, size)
        )
        rerun_viz.log_sample(
            f"val_viz/sample_{j}",
            s["occ"],
            s["start"],
            s["goal"],
            s["optimal"],
            predicted_actions=pred,
            step=epoch,
            timeline="epoch",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    set_seed(cfg.get("seed", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, train, log = cfg["data"], cfg["training"], cfg["logging"]
    size = data["grid_size"]
    os.makedirs(log["checkpoint_dir"], exist_ok=True)
    # Train metrics are per-step, val metrics are per-epoch — different schemas,
    # so they go to separate CSVs. A single logger would lock its header to the
    # first row's keys (the train row) and silently drop the val columns.
    train_logger = CSVLogger(log["log_dir"], name="train_metrics")
    val_logger = CSVLogger(log["log_dir"], name="val_metrics")

    ddir = data["data_dir"]
    train_loader = make_loader(
        f"{ddir}_train.parquet", train["batch_size"], True, data["num_workers"]
    )
    val_loader = make_loader(
        f"{ddir}_val.parquet", train["batch_size"], False, data["num_workers"]
    )

    # optionally fix a small set of val samples to watch in rerun each epoch (Phase 4)
    n_viz = log.get("viz_samples", 0)
    viz_samples = []
    use_rerun = n_viz > 0
    if use_rerun:
        viz_samples = rerun_viz.load_viz_samples(f"{ddir}_val.parquet", n_viz)
        # save_path persists all graphs to a .rrd so they're viewable after training:
        #   rerun logs/<run>/train.rrd
        rerun_viz.init("trm-train", save_path=os.path.join(log["log_dir"], "train.rrd"))
    steps_per_epoch = len(train_loader)  # for a monotonic global step on the rerun x-axis

    model = build_model(cfg, device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = AdamW(
        model.parameters(), lr=train["lr"], weight_decay=train["weight_decay"]
    )

    ema = EMA(model, decay=train.get("ema_decay", 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=train["num_epochs"])

    best_success = -1.0
    for epoch in range(train["num_epochs"]):
        model.train()
        for step, (grids, targets) in enumerate(train_loader):
            grids, targets = grids.to(device), targets.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(grids)
                result = compute_loss(
                    out,
                    targets,
                    train.get("halt_loss_weight", 1.0),
                    train.get("deep_supervision", True),
                )
            loss = result["total_loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train["grad_clip"])
            optimizer.step()
            ema.update()

            if step % log["log_every_n_steps"] == 0:
                train_logger.log(
                    {
                        "epoch": epoch,
                        "step": step,
                        "loss": loss.item(),
                        "train_acc": result["final_acc"],
                    }
                )
                if use_rerun:
                    # global step keeps the rerun x-axis monotonic across epochs;
                    # lr is read here (before scheduler.step()) so it matches this step
                    rerun_viz.log_scalars(
                        {
                            "train/loss": loss.item(),
                            "train/acc": result["final_acc"],
                            "train/lr": scheduler.get_last_lr()[0],
                        },
                        step=epoch * steps_per_epoch + step,
                        timeline="step",
                    )

        scheduler.step()

        # evaluate and optionally visualize on the smooth EMA weights
        ema.apply_shadow()
        val = evaluate(model, val_loader, size, device)
        if viz_samples:
            _log_viz_epoch(model, viz_samples, epoch, device, size)
        ema.restore()

        print(
            f"epoch {epoch + 1}/{train['num_epochs']} | success {val['success_rate']:.3f} | "
            f"opt {val['optimality_ratio']:.3f} | cellacc {val['per_cell_acc']:.3f}"
        )
        val_logger.log({"epoch": epoch, **{f"val_{k}": v for k, v in val.items()}})
        if use_rerun:
            rerun_viz.log_scalars(
                {f"val/{k}": v for k, v in val.items()}, step=epoch, timeline="epoch"
            )

        if val["success_rate"] > best_success:
            best_success = val["success_rate"]
            # save the EMA weights as the checkpoint — they generalise better than live weights
            torch.save(
                {"state_dict": ema.shadow, "config": cfg, "epoch": epoch},
                os.path.join(log["checkpoint_dir"], "best.pt"),
            )

    train_logger.close()
    val_logger.close()
    print(f"best val success rate: {best_success:.3f}")


if __name__ == "__main__":
    main()
