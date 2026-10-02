"""
Chunking policy network — same MLP shape, but the output is K*6 instead
of 6. It predicts a whole chunk of K future actions from one observation.

(ACT uses a transformer here; we keep an MLP for clarity. The chunking
idea is what matters — the architecture is secondary at this scale.)
"""

import torch.nn as nn


class ChunkPolicy(nn.Module):
    def __init__(self, obs_dim=18, action_dim=6, chunk_size=50, hidden=512):
        super().__init__()
        self.K = chunk_size
        self.action_dim = action_dim
        out_dim = action_dim * chunk_size
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, obs):
        return self.net(obs)   # (B, K*6)
