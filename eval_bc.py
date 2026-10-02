"""
Evaluate the trained BC policy in the env — measure success rate.

This is the moment of truth: does the learned policy actually pick and
place the cube, or did it just memorize? We use FRESH seeds (different
from the ones used to collect demos) to test generalization.

Usage:
    python eval_bc.py            # headless, 30 episodes
    python eval_bc.py --render   # watch it
"""

import sys
import numpy as np
from so101_env import SO101PickPlaceEnv
from bc_policy import BCController


def main():
    render = "--render" in sys.argv
    n_episodes = 30

    env    = SO101PickPlaceEnv(
        render_mode="human" if render else None,
        max_steps=2500,
        randomize_cube=True,
    )
    policy = BCController(model_path="bc_policy.pt", stats_path="bc_stats.npz")

    # Use seeds 1000+ — DIFFERENT from the 0-72 used for demo collection
    success = 0
    for ep in range(n_episodes):
        seed = 1000 + ep
        obs, info = env.reset(seed=seed)
        policy.reset()

        for t in range(env.max_steps):
            action = policy.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            if render:
                env.render()
            if terminated or truncated:
                break

        ok = info["success"]
        success += int(ok)
        print(f"Episode {ep:3d} (seed {seed}): "
              f"{'OK  ' if ok else 'FAIL'}  steps={t+1:4d}  "
              f"cube_end={info['cube_pos'][:2].round(3)}")

    print(f"\n--- BC policy success rate: {success}/{n_episodes} "
          f"= {100*success/n_episodes:.1f}% ---")

    env.close()


if __name__ == "__main__":
    main()
