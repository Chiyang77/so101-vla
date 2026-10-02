"""
Trained chunking policy wrapped as a controller (same act() interface).

Run-time strategy — RECEDING HORIZON:
  1. Query the model -> get K predicted actions
  2. Execute the first `exec_horizon` of them
  3. Re-query from the new observation, repeat

exec_horizon < K gives closed-loop correction (re-observe often) while
still benefiting from chunking (commit to an intent for several steps).
  - exec_horizon = 1  -> closed-loop every step (most reactive, but you
                          lose most of the chunking benefit)
  - exec_horizon = K  -> fully open-loop per chunk (max chunking, least
                          reactive; can drift)
A middle value (e.g. K//2) is the usual sweet spot.
"""

import numpy as np
import torch

from chunk_model import ChunkPolicy


class ChunkController:
    def __init__(self, model_path="chunk_policy.pt", stats_path="chunk_stats.npz",
                 exec_horizon=None, ensemble=False, m=0.01, device="cpu"):
        self.device = device

        stats = np.load(stats_path)
        self.obs_mean = stats["obs_mean"].astype(np.float32)
        self.obs_std  = stats["obs_std"].astype(np.float32)
        self.K        = int(stats["chunk_size"])

        # Two run modes:
        #   ensemble=False -> receding horizon (execute exec_horizon, then replan)
        #   ensemble=True  -> temporal ensembling (ACT-style, query every step,
        #                     average all chunk predictions that target this step)
        self.ensemble = ensemble
        self.m = m   # temporal-ensemble weight decay (ACT default 0.01)
        self.exec_horizon = exec_horizon if exec_horizon is not None else self.K // 2

        self.model = ChunkPolicy(obs_dim=18, action_dim=6,
                                 chunk_size=self.K, hidden=512).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.reset()

    def reset(self):
        self._buffer = None     # receding-horizon mode
        self._idx = 0
        self._chunks = []       # temporal-ensemble mode: list of (created_step, chunk)
        self._t = 0

    def _predict_chunk(self, obs):
        obs_norm = (obs - self.obs_mean) / self.obs_std
        with torch.no_grad():
            x = torch.from_numpy(obs_norm).float().unsqueeze(0).to(self.device)
            out = self.model(x).squeeze(0).cpu().numpy()        # (K*6,)
        return out.reshape(self.K, 6)                           # (K, 6)

    def act(self, obs, info=None):
        if self.ensemble:
            return self._act_ensemble(obs)
        return self._act_receding(obs)

    def _act_receding(self, obs):
        if self._buffer is None or self._idx >= self.exec_horizon:
            self._buffer = self._predict_chunk(obs)
            self._idx = 0
        action = self._buffer[self._idx]
        self._idx += 1
        return action.astype(np.float32)

    def _act_ensemble(self, obs):
        # Query a fresh chunk every step
        chunk = self._predict_chunk(obs)
        self._chunks.append((self._t, chunk))
        # Drop chunks too old to predict the current step
        self._chunks = [(s, c) for (s, c) in self._chunks if self._t - s < self.K]

        # Gather every stored prediction that targets the current step t
        preds, weights = [], []
        for (s, c) in self._chunks:
            age = self._t - s            # 0 = freshest
            preds.append(c[age])
            weights.append(np.exp(-self.m * age))
        preds   = np.array(preds)        # (n, 6)
        weights = np.array(weights)      # (n,)
        weights /= weights.sum()
        action = (weights[:, None] * preds).sum(axis=0)

        self._t += 1
        return action.astype(np.float32)
