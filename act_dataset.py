"""
Dataset for ACT: like the chunk dataset (obs -> next K actions), but we
normalize BOTH observations and actions.

Why normalize actions too: the gripper joint target ranges roughly
[-0.17, 1.74] while some arm joints span ~[-1.9, 1.9]. L1 loss would
otherwise weight the big-range joints more. Normalizing puts every action
dimension on equal footing. The policy denormalizes at run time.
"""

import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class ACTDataset(Dataset):
    def __init__(self, demos_dir="demos", chunk_size=50, decimation=4):
        # decimation: DOWNSAMPLE the control timeline by this factor.
        # Demos are recorded at 200 Hz; decimation=4 -> 50 Hz (the ACT-paper
        # rate). This is THE fix for the copycat stall: at 50 Hz each step's
        # action differs meaningfully from the current pose, so "predict the
        # present" is no longer a near-zero-loss shortcut. The chunk is built
        # on the DECIMATED timeline (each chunk action is `decimation` physics
        # steps apart), and the policy is run at the same decimated rate.
        self.K = chunk_size
        self.decimation = decimation
        files = sorted(glob.glob(f"{demos_dir}/*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz demos found in {demos_dir}/")

        self.obs_list, self.chunk_list = [], []
        all_obs, all_act = [], []

        for f in files:
            d = np.load(f)
            obs = d["observations"].astype(np.float32)[::decimation]   # 50 Hz
            act = d["actions"].astype(np.float32)[::decimation]         # 50 Hz
            T = obs.shape[0]
            all_obs.append(obs); all_act.append(act)

            for t in range(T):
                end = min(t + self.K, T)
                chunk = act[t:end]
                if chunk.shape[0] < self.K:
                    pad = np.repeat(chunk[-1:], self.K - chunk.shape[0], axis=0)
                    chunk = np.concatenate([chunk, pad], axis=0)
                self.obs_list.append(obs[t])              # (18,)
                self.chunk_list.append(chunk)             # (K, 6)

        all_obs = np.concatenate(all_obs, axis=0)
        all_act = np.concatenate(all_act, axis=0)

        self.obs_mean = all_obs.mean(0); self.obs_std = all_obs.std(0) + 1e-6
        self.act_mean = all_act.mean(0); self.act_std = all_act.std(0) + 1e-6

        print(f"Loaded {len(files)} demos -> {len(self.obs_list):,} samples "
              f"(chunk_size={self.K})")

    def __len__(self):
        return len(self.obs_list)

    def __getitem__(self, idx):
        obs   = (self.obs_list[idx] - self.obs_mean) / self.obs_std
        chunk = (self.chunk_list[idx] - self.act_mean) / self.act_std
        return torch.from_numpy(obs).float(), torch.from_numpy(chunk).float()

    def save_stats(self, path="act_stats.npz"):
        np.savez(path, obs_mean=self.obs_mean, obs_std=self.obs_std,
                 act_mean=self.act_mean, act_std=self.act_std,
                 chunk_size=np.int32(self.K),
                 decimation=np.int32(self.decimation))
        print(f"Saved stats -> {path}")


if __name__ == "__main__":
    ds = ACTDataset(chunk_size=50)
    o, c = ds[0]
    print(f"obs {o.shape}, chunk {c.shape}")
