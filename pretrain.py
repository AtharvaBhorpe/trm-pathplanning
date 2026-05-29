"""
Main training entry. Config-driven so every experiment is one YAML diff.

    uv run python pretrain.py --config configs/base.yaml
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.loader import make_loader
from models.trm import TinyRecursiveModel
from models import cnn_baseline
from utils.ema import EMA
from utils.logging import CSVLogger
from utils import metrics as M

IGNORE_INDEX = -100

def compute_trm_loss(output: dict, targets: torch.LongTensor, halt_weight=1.0) -> dict:
    """
    Deep-supervision loss = sum over T supervision steps of
        (per-cell cross-entropy) + halt_weight * (halting BCE).

    Args:
        output: dict from the model with 'logits' (list of T tensors (B,L,C))
                and 'halt_probs' (list of T tensors (B,1), possibly empty).
        targets: LongTensor (B, L) of action ids; masked cells == IGNORE_INDEX.
        halt_weight: lambda on the halting term.

    Returns:
        dict with 'total_loss' (scalar tensor) and 'final_acc' (float, for logging).

    Steps (TODO):
      for each step t:
        ce_t = F.cross_entropy(logits[t].reshape(-1, C), targets.reshape(-1),
                               ignore_index=IGNORE_INDEX)
        if halting:
            # q = 1 if this step's per-cell prediction is "correct enough", else 0.
            # A reasonable target: per-sample fraction of correct non-masked cells
            # thresholded, OR just the mean per-cell correctness. Detach it.
            # halt_bce = F.binary_cross_entropy(halt_probs[t].squeeze(-1), q)
            ...
        step_loss = ce_t + halt_weight * halt_bce
      total = sum(step_loss)
    """
    total_loss = 0.0
    for t, logits in enumerate(output['logits']):
        ce_t = F.cross_entropy(logits[t].reshape(-1, logits[t].shape[-1]), targets.reshape(-1),
                               ignore_index=IGNORE_INDEX)
        if 'halt_probs' in output and len(output['halt_probs']) > t:
            halt_prob = output['halt_probs'][t].squeeze(-1)  # (B,)
            with torch.no_grad():
                pred_actions = logits[t].argmax(dim=-1)  # (B, L)
                correct = (pred_actions == targets) & (targets != IGNORE_INDEX)  # (B, L)
                q = correct.float().mean(dim=1)  # (B,) fraction of correct non-masked cells
            halt_bce = F.binary_cross_entropy(halt_prob, q)
        else:
            halt_bce = 0.0
        step_loss = ce_t + halt_weight * halt_bce
        total_loss += step_loss
    return {
        'total_loss': total_loss,
        'final_acc': M.per_cell_accuracy(output['logits'][-1].argmax(dim=-1), targets, IGNORE_INDEX)
    }

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_model(cfg, device):
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
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, size, device):
    model.eval()
    agg = {"success_rate": [], "optimality_ratio": [], "per_cell_acc": []}
    for grids, targets in loader:
        grids = grids.to(device)
        out = model(grids)
        preds = out["final_logits"].argmax(dim=-1)            # (B, L)
        # recover flattened start/goal from the marker tokens in the input
        starts = (grids == 2).float().argmax(dim=1).cpu().numpy()
        goals = (grids == 3).float().argmax(dim=1).cpu().numpy()
        b = M.evaluate_batch(grids, preds, starts, goals, size)
        agg["success_rate"].append(b["success_rate"])
        agg["optimality_ratio"].append(b["optimality_ratio"])
        agg["per_cell_acc"].append(M.per_cell_accuracy(preds.cpu(), targets, IGNORE_INDEX))
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t, lg = cfg["training"], cfg["logging"]
    size = cfg["data"]["grid_size"]
    ddir = cfg["data"]["data_dir"]
    os.makedirs(lg["checkpoint_dir"], exist_ok=True)
    logger = CSVLogger(lg["log_dir"])

    train_loader = make_loader(f"{ddir}_train.parquet", t["batch_size"], True, cfg["data"]["num_workers"])
    val_loader = make_loader(f"{ddir}_val.parquet", t["batch_size"], False, cfg["data"]["num_workers"])

    model = build_model(cfg, device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    opt = AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = CosineAnnealingLR(opt, T_max=t["num_epochs"])
    ema = EMA(model, t["ema_decay"]) if t.get("use_ema", True) else None

    amp = t["mixed_precision"]
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    scaler = torch.amp.GradScaler(enabled=(amp == "fp16"))
    accum = t["grad_accum_steps"]

    best_success, patience = -1.0, 0
    for epoch in range(t["num_epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, (grids, targets) in enumerate(train_loader):
            grids, targets = grids.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(grids)
                loss_dict = compute_trm_loss(out, targets, t.get("halt_loss_weight", 1.0))
                loss = loss_dict["total_loss"] / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if ema: ema.update()
            if step % lg["log_every_n_steps"] == 0:
                logger.log({"epoch": epoch, "step": step, "loss": loss_dict["total_loss"].item()})
        sched.step()

        if ema: ema.apply_shadow()
        val = evaluate(model, val_loader, size, device)
        if ema: ema.restore()
        print(f"epoch {epoch+1}/{t['num_epochs']} | success {val['success_rate']:.3f} | "
              f"opt {val['optimality_ratio']:.3f} | cellacc {val['per_cell_acc']:.3f}")
        logger.log({"epoch": epoch, **{f"val_{k}": v for k, v in val.items()}})

        if val["success_rate"] > best_success:
            best_success = val["success_rate"]; patience = 0
            sd = ema.shadow if ema else model.state_dict()
            torch.save({"state_dict": sd, "config": cfg, "epoch": epoch},
                       os.path.join(lg["checkpoint_dir"], "best.pt"))
        else:
            patience += 1
            if patience >= t["early_stopping_patience"]:
                print(f"early stopping at epoch {epoch+1}"); break

    logger.close()
    print(f"best val success rate: {best_success:.3f}")


if __name__ == "__main__":
    main()
