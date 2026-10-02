"""
Train the action-chunking policy. Same supervised loop as bc_train.py,
but the target is a flattened chunk of K actions (K*6 numbers).

Usage:
    python chunk_train.py            # uses CUDA if it works
    python chunk_train.py --cpu      # force CPU (RTX 5080 / CUDA-13 issue)
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from chunk_dataset import ChunkDataset
from chunk_model import ChunkPolicy


# ---- Config ----
CHUNK_SIZE = 50
EPOCHS     = 150
BATCH_SIZE = 256
LR         = 1e-3
HIDDEN     = 512
VAL_FRAC   = 0.1
MODEL_OUT  = "chunk_policy.pt"
STATS_OUT  = "chunk_stats.npz"


def main():
    force_cpu = "--cpu" in sys.argv
    device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}\n")

    full = ChunkDataset(demos_dir="demos", chunk_size=CHUNK_SIZE, normalize=True)
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

    model = ChunkPolicy(obs_dim=18, action_dim=6,
                        chunk_size=CHUNK_SIZE, hidden=HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}\n")

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for obs, chunk in train_loader:
            obs, chunk = obs.to(device), chunk.to(device)
            loss = loss_fn(model(obs), chunk)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * obs.size(0)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, chunk in val_loader:
                obs, chunk = obs.to(device), chunk.to(device)
                val_loss += loss_fn(model(obs), chunk).item() * obs.size(0)
        val_loss /= n_val

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
