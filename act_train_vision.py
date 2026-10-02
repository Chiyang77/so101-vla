"""
Train the vision ACT: image + proprio -> K-action chunk.

Loss = L1(chunk) + beta*KL(z||N(0,1))  (same CVAE objective as state ACT).

Adds a cheap DrQ-style random-shift image augmentation on the training batch
(pad by a few px, random crop back). Vision policies overfit small demo sets
fast; this is the single most effective, near-free regularizer.

Usage:
    python act_train_vision.py --epochs 120 --batch 128 --beta 10 --patience 15
    python act_train_vision.py --cpu
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from act_dataset_vision import VisionACTDataset
from act_model_vision import VisionACTPolicy, kl_divergence


def get_arg(flag, default, cast):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def random_shift(imgs, pad=4):
    """DrQ random shift: reflect-pad then random-crop back to original size."""
    B, C, H, W = imgs.shape
    imgs = F.pad(imgs, (pad, pad, pad, pad), mode="replicate")
    # one random crop offset per image
    ox = torch.randint(0, 2 * pad + 1, (B,), device=imgs.device)
    oy = torch.randint(0, 2 * pad + 1, (B,), device=imgs.device)
    out = torch.empty(B, C, H, W, device=imgs.device, dtype=imgs.dtype)
    for i in range(B):
        out[i] = imgs[i, :, oy[i]:oy[i] + H, ox[i]:ox[i] + W]
    return out


def main():
    force_cpu = "--cpu" in sys.argv
    EPOCHS   = get_arg("--epochs", 120, int)
    BATCH    = get_arg("--batch", 128, int)
    LR       = get_arg("--lr", 1e-4, float)
    BETA     = get_arg("--beta", 10.0, float)
    CHUNK    = get_arg("--chunk", 50, int)
    D_MODEL  = get_arg("--dmodel", 256, int)
    PATIENCE = get_arg("--patience", 15, int)
    AUG      = "--no-aug" not in sys.argv

    device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}  epochs={EPOCHS} batch={BATCH} beta={BETA} "
          f"chunk={CHUNK} aug={AUG}\n")

    full = VisionACTDataset(demos_dir="demos_vision", chunk_size=CHUNK)
    full.save_stats("act_vision_stats.npz")
    n_val = int(len(full) * 0.1); n_train = len(full) - n_val
    tr, va = random_split(full, [n_train, n_val],
                          generator=torch.Generator().manual_seed(0))
    print(f"Train {n_train:,}  Val {n_val:,}\n")

    tl = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
    vl = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

    model = VisionACTPolicy(proprio_dim=12, action_dim=6, chunk_size=CHUNK,
                            d_model=D_MODEL).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    l1 = nn.L1Loss()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}\n")

    best = float("inf"); since = 0
    for ep in range(1, EPOCHS + 1):
        model.train(); tr_l1 = tr_kl = 0.0
        for ie, iw, prop, chunk in tl:
            ie, iw = ie.to(device), iw.to(device)
            prop, chunk = prop.to(device), chunk.to(device)
            if AUG:
                ie, iw = random_shift(ie), random_shift(iw)
            pred, mu, lv = model(ie, iw, prop, chunk)
            recon = l1(pred, chunk); kl = kl_divergence(mu, lv)
            loss = recon + BETA * kl
            opt.zero_grad(); loss.backward(); opt.step()
            tr_l1 += recon.item() * ie.size(0); tr_kl += kl.item() * ie.size(0)
        tr_l1 /= n_train; tr_kl /= n_train

        model.eval(); val = 0.0
        with torch.no_grad():
            for ie, iw, prop, chunk in vl:
                ie, iw = ie.to(device), iw.to(device)
                prop, chunk = prop.to(device), chunk.to(device)
                pred, _, _ = model(ie, iw, prop, actions=None)   # z=0
                val += l1(pred, chunk).item() * ie.size(0)
        val /= n_val

        if val < best:
            best = val; since = 0
            torch.save(model.state_dict(), "act_vision_policy.pt"); star = " *"
        else:
            since += 1; star = ""
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d}  train_L1={tr_l1:.4f} KL={tr_kl:.4f}  "
                  f"val_L1(z=0)={val:.4f}{star}")
        if since >= PATIENCE:
            print(f"\nEarly stop at epoch {ep} (no val gain for {PATIENCE}).")
            break

    print(f"\nBest val L1: {best:.4f}\nSaved -> act_vision_policy.pt")


if __name__ == "__main__":
    main()
