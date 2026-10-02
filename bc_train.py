"""
Train the behavior cloning policy.

Standard supervised learning loop:
    for each epoch:
        for each batch of (obs, action):
            pred = model(obs)
            loss = MSE(pred, action)        # how wrong were we?
            loss.backward()                  # compute gradients
            optimizer.step()                 # nudge weights to reduce loss

We hold out 10% of the data as a validation set to watch for overfitting
(train loss keeps dropping but val loss rises = memorizing, not learning).
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from bc_dataset import BCDataset
from bc_model import BCPolicy


# ---- Config ----
EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 1e-3
HIDDEN      = 256
VAL_FRAC    = 0.1
MODEL_OUT   = "bc_policy.pt"
STATS_OUT   = "bc_stats.npz"


def main():
    # Note: RTX 5080 (sm_120) needs a CUDA-13 PyTorch build. If the installed
    # torch doesn't support the GPU, pass --cpu to force CPU (fine for this MLP).
    force_cpu = "--cpu" in sys.argv
    device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}\n")

    # ---- Data ----
    full = BCDataset(demos_dir="demos", normalize=True)
    full.save_stats(STATS_OUT)

    n_val   = int(len(full) * VAL_FRAC)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(0)
    )
    print(f"Train: {n_train:,}  Val: {n_val:,}\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ---- Model + optimizer ----
    model     = BCPolicy(obs_dim=18, action_dim=6, hidden=HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}\n")

    # ---- Training loop ----
    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for obs, act in train_loader:
            obs, act = obs.to(device), act.to(device)
            pred = model(obs)
            loss = loss_fn(pred, act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * obs.size(0)
        train_loss /= n_train

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, act in val_loader:
                obs, act = obs.to(device), act.to(device)
                val_loss += loss_fn(model(obs), act).item() * obs.size(0)
        val_loss /= n_val

        # --- checkpoint best ---
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_OUT)
            star = " *"
        else:
            star = ""

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train_loss={train_loss:.5f}  "
                  f"val_loss={val_loss:.5f}{star}")

    print(f"\nBest val loss: {best_val:.5f}")
    print(f"Saved best model -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
