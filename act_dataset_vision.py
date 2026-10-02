"""
Dataset for vision ACT. Demos are already at 50 Hz (decimated during
collection), so we just build (image, proprio) -> next-K-actions chunks.

Images stay uint8 in RAM (indexed per (demo, t)) to avoid duplicating them per
sample; __getitem__ converts to float[0,1] CHW. Proprio and actions are
normalized (mean/std); the policy denormalizes actions at run time.
"""

import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class VisionACTDataset(Dataset):
    def __init__(self, demos_dir="demos_vision", chunk_size=50):
        self.K = chunk_size
        files = sorted(glob.glob(f"{demos_dir}/*.npz"))
        if not files:
            raise FileNotFoundError(f"No demos in {demos_dir}/")

        self.img_ext, self.img_wrist = [], []               # per-demo arrays
        self.proprio, self.actions = [], []
        self.index = []                                     # (demo_idx, t)
        all_prop, all_act = [], []

        for di, f in enumerate(files):
            d = np.load(f)
            ie = d["images_ext"]                       # (T,H,W,3) uint8
            iw = d["images_wrist"]                     # (T,H,W,3) uint8
            prop = d["proprio"].astype(np.float32)     # (T,12)
            act  = d["actions"].astype(np.float32)     # (T,6)
            self.img_ext.append(ie); self.img_wrist.append(iw)
            self.proprio.append(prop); self.actions.append(act)
            all_prop.append(prop); all_act.append(act)
            for t in range(ie.shape[0]):
                self.index.append((di, t))

        all_prop = np.concatenate(all_prop); all_act = np.concatenate(all_act)
        self.prop_mean = all_prop.mean(0); self.prop_std = all_prop.std(0) + 1e-6
        self.act_mean  = all_act.mean(0);  self.act_std  = all_act.std(0) + 1e-6

        print(f"Loaded {len(files)} vision demos -> {len(self.index):,} samples "
              f"(chunk={self.K}, img={self.img_ext[0].shape[1:]}, 2 cameras)")

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _chw(img_uint8):
        return np.transpose(img_uint8.astype(np.float32) / 255.0, (2, 0, 1))

    def __getitem__(self, idx):
        di, t = self.index[idx]
        ie = self._chw(self.img_ext[di][t])
        iw = self._chw(self.img_wrist[di][t])
        prop = (self.proprio[di][t] - self.prop_mean) / self.prop_std

        act = self.actions[di]
        end = min(t + self.K, act.shape[0])
        chunk = act[t:end]
        if chunk.shape[0] < self.K:
            pad = np.repeat(chunk[-1:], self.K - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        chunk = (chunk - self.act_mean) / self.act_std

        return (torch.from_numpy(ie).float(),
                torch.from_numpy(iw).float(),
                torch.from_numpy(prop).float(),
                torch.from_numpy(chunk).float())

    def save_stats(self, path="act_vision_stats.npz"):
        np.savez(path, prop_mean=self.prop_mean, prop_std=self.prop_std,
                 act_mean=self.act_mean, act_std=self.act_std,
                 chunk_size=np.int32(self.K))
        print(f"Saved stats -> {path}")


if __name__ == "__main__":
    ds = VisionACTDataset()
    ie, iw, prop, chunk = ds[0]
    print("img_ext", ie.shape, "img_wrist", iw.shape,
          "proprio", prop.shape, "chunk", chunk.shape)
