"""
Behavior Cloning dataset: load all demo .npz files and expose every
(observation, action) pair as a flat PyTorch dataset.

Behavior cloning treats imitation as plain supervised regression:
    input  = observation (18-dim)
    target = action the expert took (6-dim)
We don't care about episode boundaries for vanilla BC — we just pool
ALL (obs, action) pairs from all demos into one big bag and learn the
mapping obs -> action.

We also compute normalization stats (mean/std) over the observations.
Neural networks train much better when inputs are roughly zero-mean,
unit-variance. We save these stats so the policy can normalize at eval.
"""

import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class BCDataset(Dataset):
    def __init__(self, demos_dir="demos", normalize=True):
        files = sorted(glob.glob(f"{demos_dir}/*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz demos found in {demos_dir}/")

        obs_chunks, act_chunks = [], []
        for f in files:
            d = np.load(f)
            obs_chunks.append(d["observations"])   # (T, 18)
            act_chunks.append(d["actions"])         # (T, 6)

        # Pool every step from every demo into one big array
        self.obs = np.concatenate(obs_chunks, axis=0).astype(np.float32)  # (N, 18)
        self.act = np.concatenate(act_chunks, axis=0).astype(np.float32)  # (N, 6)

        print(f"Loaded {len(files)} demos -> {self.obs.shape[0]:,} (obs, action) pairs")

        # --- Normalization stats over observations ---
        if normalize:
            self.obs_mean = self.obs.mean(axis=0)
            self.obs_std  = self.obs.std(axis=0) + 1e-6   # avoid divide-by-zero
        else:
            self.obs_mean = np.zeros(self.obs.shape[1], dtype=np.float32)
            self.obs_std  = np.ones(self.obs.shape[1], dtype=np.float32)

        # Pre-normalize the stored observations
        self.obs_norm = (self.obs - self.obs_mean) / self.obs_std

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs_norm[idx]),
            torch.from_numpy(self.act[idx]),
        )

    def save_stats(self, path="bc_stats.npz"):
        np.savez(path, obs_mean=self.obs_mean, obs_std=self.obs_std)
        print(f"Saved normalization stats -> {path}")


if __name__ == "__main__":
    # Quick self-test
    ds = BCDataset()
    print(f"Dataset length: {len(ds)}")
    o, a = ds[0]
    print(f"Sample obs shape: {o.shape}, action shape: {a.shape}")
    print(f"obs mean range: [{ds.obs_mean.min():.3f}, {ds.obs_mean.max():.3f}]")
    print(f"obs std  range: [{ds.obs_std.min():.3f}, {ds.obs_std.max():.3f}]")
