"""
Train the minimal ACT.

Loss = L1(predicted chunk, expert chunk) + beta * KL(z || N(0,1))

The KL term is the CVAE regularizer: it keeps the latent z close to a
standard normal so that at test time we can just set z = 0. beta controls
how strongly — too high and z collapses to noise (decoder ignores it,
back to averaging); too low and z memorizes (poor test behavior). ACT
uses a small beta (~10) relative to reconstruction; we tune by feel.

Usage:
    python act_train.py            # CUDA if available & working
    python act_train.py --cpu      # force CPU
    python act_train.py --epochs 50 --batch 64
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from act_dataset import ACTDataset
from act_model import ACTPolicy, kl_divergence


def get_arg(flag, default, cast):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


def main():
    force_cpu  = "--cpu" in sys.argv
    EPOCHS     = get_arg("--epochs", 100, int)
    BATCH_SIZE = get_arg("--batch", 256, int)
    LR         = get_arg("--lr", 1e-4, float)
    CHUNK_SIZE = get_arg("--chunk", 50, int)
    BETA       = get_arg("--beta", 10.0, float)
    D_MODEL    = get_arg("--dmodel", 256, int)
    DECIM      = get_arg("--decim", 4, int)        # 200 Hz / 4 = 50 Hz control
    PATIENCE   = get_arg("--patience", 15, int)   # stop if val_L1 doesn't
                                                  # improve for this many epochs

    device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}  epochs={EPOCHS} batch={BATCH_SIZE} "
          f"lr={LR} beta={BETA} d_model={D_MODEL}\n")

    full = ACTDataset(demos_dir="demos", chunk_size=CHUNK_SIZE, decimation=DECIM)
    full.save_stats("act_stats.npz")

    n_val   = int(len(full) * 0.1)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    print(f"Train: {n_train:,}  Val: {n_val:,}\n")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    obs_dim = full.obs_list[0].shape[0]   # 18 (base) or 20 (goal-conditioned)
    print(f"obs_dim = {obs_dim}")
    model = ACTPolicy(obs_dim=obs_dim, action_dim=6, chunk_size=CHUNK_SIZE,
                      d_model=D_MODEL).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    l1 = nn.L1Loss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}\n")

    best_val = float("inf")
    epochs_since_best = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_l1 = tr_kl = 0.0
        for obs, chunk in train_loader:
            obs, chunk = obs.to(device), chunk.to(device)
            pred, mu, logvar = model(obs, chunk)
            recon = l1(pred, chunk)
            kl    = kl_divergence(mu, logvar)
            loss  = recon + BETA * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_l1 += recon.item() * obs.size(0)
            tr_kl += kl.item()   * obs.size(0)
        tr_l1 /= n_train; tr_kl /= n_train

        # Validation reconstruction uses z=0 (the test-time path)
        model.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for obs, chunk in val_loader:
                obs, chunk = obs.to(device), chunk.to(device)
                pred, _, _ = model(obs, actions=None)   # z = 0
                val_l1 += l1(pred, chunk).item() * obs.size(0)
        val_l1 /= n_val

        if val_l1 < best_val:
            best_val = val_l1
            epochs_since_best = 0
            torch.save(model.state_dict(), "act_policy.pt")
            star = " *"
        else:
            epochs_since_best += 1
            star = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train_L1={tr_l1:.4f}  train_KL={tr_kl:.4f}  "
                  f"val_L1(z=0)={val_l1:.4f}{star}")

        if epochs_since_best >= PATIENCE:
            print(f"\nEarly stop at epoch {epoch}: no val improvement for "
                  f"{PATIENCE} epochs.")
            break

    print(f"\nBest val L1: {best_val:.4f}")
    print("Saved best model -> act_policy.pt")


if __name__ == "__main__":
    main()
