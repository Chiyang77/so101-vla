"""
Action-chunking dataset.

Vanilla BC learned:  obs[t]  ->  action[t]          (one action)
Chunking learns:     obs[t]  ->  action[t : t+K]    (K future actions)

Predicting a CHUNK of future actions from one observation is the core
idea of ACT (Action Chunking Transformer). It helps two ways:
  1. Fewer decision points at run time -> errors compound K times less
  2. A chunk commits to ONE intent ("descend then close") instead of
     averaging conflicting single-step actions (the multimodality fix)

We must NOT let a chunk cross an episode boundary. Near the end of an
episode we pad by repeating the last action.
"""

import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class ChunkDataset(Dataset):
    def __init__(self, demos_dir="demos", chunk_size=50, normalize=True):
        self.K = chunk_size
        files = sorted(glob.glob(f"{demos_dir}/*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz demos found in {demos_dir}/")

        self.samples = []   # list of (obs, action_chunk) pairs
        all_obs = []        # for computing normalization stats

        for f in files:
            d = np.load(f)
            obs = d["observations"].astype(np.float32)   # (T, 18)
            act = d["actions"].astype(np.float32)         # (T, 6)
            T = obs.shape[0]
            all_obs.append(obs)

            for t in range(T):
                # Grab actions[t : t+K], padding with the last action if needed
                end = min(t + self.K, T)
                chunk = act[t:end]                        # (<=K, 6)
                if chunk.shape[0] < self.K:
                    pad = np.repeat(chunk[-1:], self.K - chunk.shape[0], axis=0)
                    chunk = np.concatenate([chunk, pad], axis=0)
                self.samples.append((obs[t], chunk.reshape(-1)))  # chunk flat -> (K*6,)

        all_obs = np.concatenate(all_obs, axis=0)
        print(f"Loaded {len(files)} demos -> {len(self.samples):,} chunk samples "
              f"(chunk_size={self.K})")

        # Normalization stats over observations
        if normalize:
            self.obs_mean = all_obs.mean(axis=0)
            self.obs_std  = all_obs.std(axis=0) + 1e-6
        else:
            self.obs_mean = np.zeros(all_obs.shape[1], dtype=np.float32)
            self.obs_std  = np.ones(all_obs.shape[1], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, chunk = self.samples[idx]
        obs_norm = (obs - self.obs_mean) / self.obs_std
        return torch.from_numpy(obs_norm).float(), torch.from_numpy(chunk).float()

    def save_stats(self, path="chunk_stats.npz"):
        np.savez(path, obs_mean=self.obs_mean, obs_std=self.obs_std,
                 chunk_size=np.int32(self.K))
        print(f"Saved stats -> {path}")


if __name__ == "__main__":
    ds = ChunkDataset(chunk_size=50)
    o, c = ds[0]
    print(f"obs shape: {o.shape}, chunk shape: {c.shape}  (= K*6 = {ds.K*6})")
