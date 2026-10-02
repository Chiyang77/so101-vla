"""
Trained ACT wrapped as a controller (same act() interface as the others).

Run-time: z = 0 (prior mean) + temporal ensembling, exactly like ACT's
default inference. Query every step, average overlapping chunk predictions
for the current timestep (recency-weighted).
"""

import numpy as np
import torch

from act_model import ACTPolicy


class ACTController:
    def __init__(self, model_path="act_policy.pt", stats_path="act_stats.npz",
                 d_model=256, ensemble=True, m=0.01, lookahead=0, device="cpu"):
        self.device = device
        self.ensemble = ensemble
        self.m = m
        # lookahead>0: command chunk[lookahead] (a near-FUTURE target) every step,
        # instead of chunk[0] (~current pose). Breaks the copycat stall by giving
        # the position controller a target with a meaningful offset.
        self.lookahead = lookahead

        stats = np.load(stats_path)
        self.obs_mean = stats["obs_mean"].astype(np.float32)
        self.obs_std  = stats["obs_std"].astype(np.float32)
        self.act_mean = stats["act_mean"].astype(np.float32)
        self.act_std  = stats["act_std"].astype(np.float32)
        self.K        = int(stats["chunk_size"])
        self.decimation = int(stats["decimation"]) if "decimation" in stats else 1
        obs_dim = len(self.obs_mean)   # 18 (base) or 20 (goal-conditioned)

        self.model = ACTPolicy(obs_dim=obs_dim, action_dim=6, chunk_size=self.K,
                               d_model=d_model).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.reset()

    def reset(self):
        self._chunks = []   # list of (created_step, chunk) for temporal ensembling
        self._t = 0
        self._buffer = None
        self._idx = 0

    def _predict_chunk(self, obs):
        obs_norm = (obs - self.obs_mean) / self.obs_std
        with torch.no_grad():
            x = torch.from_numpy(obs_norm).float().unsqueeze(0).to(self.device)
            pred, _, _ = self.model(x, actions=None)        # z = 0
            chunk_norm = pred.squeeze(0).cpu().numpy()      # (K, 6) normalized
        return chunk_norm * self.act_std + self.act_mean    # denormalize

    def act(self, obs, info=None):
        if self.lookahead > 0:
            # Predict a fresh chunk every step, command the lookahead-th action.
            chunk = self._predict_chunk(obs)
            k = min(self.lookahead, self.K - 1)
            return chunk[k].astype(np.float32)

        if not self.ensemble:
            # simple: predict a chunk, execute fully, replan
            if self._buffer is None or self._idx >= self.K:
                self._buffer = self._predict_chunk(obs)
                self._idx = 0
            a = self._buffer[self._idx]; self._idx += 1
            return a.astype(np.float32)

        # temporal ensembling
        chunk = self._predict_chunk(obs)
        self._chunks.append((self._t, chunk))
        self._chunks = [(s, c) for (s, c) in self._chunks if self._t - s < self.K]

        preds, weights = [], []
        for (s, c) in self._chunks:
            age = self._t - s
            preds.append(c[age])
            weights.append(np.exp(-self.m * age))
        preds = np.array(preds); weights = np.array(weights)
        weights /= weights.sum()
        action = (weights[:, None] * preds).sum(axis=0)

        self._t += 1
        return action.astype(np.float32)
