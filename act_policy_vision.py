"""
Trained vision ACT wrapped as a controller. Takes obs = {image, proprio},
runs z=0 inference + temporal ensembling, returns a 6-D action.
"""

import numpy as np
import torch
from act_model_vision import VisionACTPolicy


class VisionACTController:
    def __init__(self, model_path="act_vision_policy.pt",
                 stats_path="act_vision_stats.npz", d_model=256,
                 ensemble=True, m=0.01, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ensemble = ensemble
        self.m = m

        s = np.load(stats_path)
        self.prop_mean = s["prop_mean"].astype(np.float32)
        self.prop_std  = s["prop_std"].astype(np.float32)
        self.act_mean  = s["act_mean"].astype(np.float32)
        self.act_std   = s["act_std"].astype(np.float32)
        self.K         = int(s["chunk_size"])

        self.model = VisionACTPolicy(proprio_dim=12, action_dim=6,
                                     chunk_size=self.K, d_model=d_model).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.reset()

    def reset(self):
        self._chunks = []   # (created_step, chunk) for temporal ensembling
        self._t = 0

    @staticmethod
    def _chw(img):
        return np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))[None]

    def _predict_chunk(self, img_ext, img_wrist, proprio_raw):
        prop = ((proprio_raw - self.prop_mean) / self.prop_std)[None]
        with torch.no_grad():
            ie = torch.from_numpy(self._chw(img_ext)).float().to(self.device)
            iw = torch.from_numpy(self._chw(img_wrist)).float().to(self.device)
            pt = torch.from_numpy(prop).float().to(self.device)
            pred, _, _ = self.model(ie, iw, pt, actions=None)  # z=0
            chunk = pred.squeeze(0).cpu().numpy()              # (K,6) normalized
        return chunk * self.act_std + self.act_mean            # denormalize

    def act(self, obs, info=None):
        chunk = self._predict_chunk(obs["image_ext"], obs["image_wrist"],
                                    obs["proprio"])
        if not self.ensemble:
            a = chunk[0]
            self._t += 1
            return a.astype(np.float32)

        self._chunks.append((self._t, chunk))
        self._chunks = [(s, c) for (s, c) in self._chunks if self._t - s < self.K]
        preds, w = [], []
        for (s, c) in self._chunks:
            age = self._t - s
            preds.append(c[age]); w.append(np.exp(-self.m * age))
        preds = np.array(preds); w = np.array(w); w /= w.sum()
        action = (w[:, None] * preds).sum(0)
        self._t += 1
        return action.astype(np.float32)
