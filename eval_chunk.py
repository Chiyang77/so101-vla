"""
Evaluate the chunking policy on FRESH seeds (1000+), same protocol as
eval_bc.py so the numbers are directly comparable.

Usage:
    python eval_chunk.py                  # headless, 30 episodes
    python eval_chunk.py --render         # watch it
    python eval_chunk.py --horizon 25     # override exec_horizon
"""

import sys
import numpy as np
from so101_env import SO101PickPlaceEnv
from chunk_policy import ChunkController


def main():
    render = "--render" in sys.argv
    ensemble = "--ensemble" in sys.argv
    horizon = None
    if "--horizon" in sys.argv:
        horizon = int(sys.argv[sys.argv.index("--horizon") + 1])

    n_episodes = 30
    env = SO101PickPlaceEnv(
        render_mode="human" if render else None,
        max_steps=2500,
        randomize_cube=True,
    )
    policy = ChunkController(exec_horizon=horizon, ensemble=ensemble)
    print(f"chunk_size={policy.K}  exec_horizon={policy.exec_horizon}\n")

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

    print(f"\n--- Chunk policy success rate: {success}/{n_episodes} "
          f"= {100*success/n_episodes:.1f}% ---")
    env.close()


if __name__ == "__main__":
    main()
