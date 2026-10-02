"""
The trained BC policy wrapped as a controller — same interface as
ScriptedPickPlaceController, so it drops straight into the env loop.

    controller.reset()
    action = controller.act(obs, info)

The contrast with the scripted expert:
  - Scripted: runs IK + state machine, needs privileged model access
  - Learned : just a forward pass through the MLP on the observation
The learned policy has NO idea about IK, states, or the cube site —
it only knows "this observation -> this action" from training.
"""

import numpy as np
import torch

from bc_model import BCPolicy


class BCController:
    def __init__(self, model_path="bc_policy.pt", stats_path="bc_stats.npz",
                 device=None):
        # Default to CPU — the tiny MLP runs instantly on CPU, and avoids the
        # RTX 5080 sm_120 / CUDA-13 compatibility issue with this torch build.
        self.device = device or "cpu"

        # Load model weights
        self.model = BCPolicy(obs_dim=18, action_dim=6, hidden=256).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Load normalization stats (must match what training used)
        stats = np.load(stats_path)
        self.obs_mean = stats["obs_mean"].astype(np.float32)
        self.obs_std  = stats["obs_std"].astype(np.float32)

    def reset(self):
        # Stateless policy — nothing to reset. (Method exists for interface parity.)
        pass

    def act(self, obs, info=None):
        # Normalize the observation exactly as during training
        obs_norm = (obs - self.obs_mean) / self.obs_std

        with torch.no_grad():
            x = torch.from_numpy(obs_norm).float().unsqueeze(0).to(self.device)
            action = self.model(x).squeeze(0).cpu().numpy()

        return action.astype(np.float32)
