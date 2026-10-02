"""
The behavior cloning policy network: a small MLP.

Maps a normalized 18-dim observation to a 6-dim action (joint targets).
This is deliberately simple — a few fully-connected layers. For a
state-based task like ours, an MLP is plenty. (Vision-based tasks need
CNNs/transformers — that comes later with LeRobot.)
"""

import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    def __init__(self, obs_dim=18, action_dim=6, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)
