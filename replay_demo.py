"""
Watch a saved demo: replay its recorded actions through the env with the
viewer open. This shows EXACTLY what was saved (deterministic via the seed),
so you can visually confirm the demo is a clean pick-lift-place.

Usage:
    python replay_demo.py                      # demos/episode_0000.npz
    python replay_demo.py demos/episode_0042.npz
    python replay_demo.py 7                     # episode_0007.npz
"""

import sys
import time
import glob
import numpy as np
from so101_env import SO101PickPlaceEnv


def resolve(arg):
    if arg is None:
        return "demos/episode_0000.npz"
    if arg.endswith(".npz"):
        return arg
    return f"demos/episode_{int(arg):04d}.npz"   # bare number


def main():
    fname = resolve(sys.argv[1] if len(sys.argv) > 1 else None)
    d = np.load(fname)
    seed = int(d["seed"]); actions = d["actions"]
    print(f"Replaying {fname}  (seed={seed}, {len(actions)} steps)")

    env = SO101PickPlaceEnv(render_mode="human", max_steps=10000,
                            randomize_cube=True, goal_conditioned=True)
    obs, info = env.reset(seed=seed)
    print(f"target = {env.drop_target_xy.round(3)}")

    start = time.time()
    for i in range(len(actions) + 60):     # +60 to watch it settle
        a = actions[i] if i < len(actions) else actions[-1]
        obs, r, term, trunc, info = env.step(a)
        env.render()
        # real-time pacing (env runs at 200 Hz)
        target_t = (i + 1) / 200.0
        elapsed = time.time() - start
        if target_t > elapsed:
            time.sleep(target_t - elapsed)

    cube = info["cube_pos"]; tgt = env.drop_target_xy
    print(f"final cube = {cube.round(3)}  place_err = "
          f"{np.linalg.norm(cube[:2]-tgt)*100:.1f} cm  success = {info['success']}")
    env.close()


if __name__ == "__main__":
    main()
