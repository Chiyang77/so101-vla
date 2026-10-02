"""
Step 2 of Phase 4: collect a dataset of (obs, action) trajectories
from the scripted expert.

For each random seed:
  - Reset env with that seed
  - Run scripted expert until done
  - If successful, save the trajectory to demos/episode_XXXX.npz
  - If failed, discard

Each saved file contains arrays for the full episode:
  observations : (T, 18)   joint state + cube + ee positions
  actions      : (T, 6)    joint targets sent to actuators
  rewards      : (T,)
  cube_starts  : (3,)      where the cube was at episode start
  seed         : ()        the random seed used
  steps        : ()        episode length

We aim for N_DEMOS successful demos. With ~73% success rate, that means
running ~N_DEMOS/0.73 episodes.
"""

import os
import time
import numpy as np
from so101_env import SO101PickPlaceEnv
from scripted_controller import ScriptedPickPlaceController


# ---- Config ----
N_DEMOS    = 200                 # how many successful demos we want
DEMOS_DIR  = "demos"             # output folder
MAX_SEEDS  = 400                 # safety cap on how many seeds to try


def run_one_episode(env, expert, seed):
    """Run one episode, return (success, obs_list, action_list, reward_list, cube_start)."""
    obs, info = env.reset(seed=seed)
    expert.reset()
    cube_start = info["cube_pos"].copy()

    obs_list, action_list, reward_list = [], [], []

    for t in range(env.max_steps):
        action = expert.act(obs, info)

        # Record BEFORE stepping (obs -> action mapping is what IL learns)
        obs_list.append(obs.copy())
        action_list.append(action.copy())

        obs, reward, terminated, truncated, info = env.step(action)
        reward_list.append(reward)

        if terminated or truncated:
            break

    return info["success"], obs_list, action_list, reward_list, cube_start


def save_demo(idx, seed, obs_list, action_list, reward_list, cube_start):
    """Save one trajectory to demos/episode_NNNN.npz."""
    fname = os.path.join(DEMOS_DIR, f"episode_{idx:04d}.npz")
    np.savez(
        fname,
        observations=np.array(obs_list, dtype=np.float32),
        actions=np.array(action_list, dtype=np.float32),
        rewards=np.array(reward_list, dtype=np.float32),
        cube_start=cube_start.astype(np.float32),
        seed=np.int32(seed),
        steps=np.int32(len(obs_list)),
    )


def main():
    os.makedirs(DEMOS_DIR, exist_ok=True)

    env    = SO101PickPlaceEnv(render_mode=None, max_steps=2500, randomize_cube=True,
                               goal_conditioned=True)
    expert = ScriptedPickPlaceController(env)

    n_saved   = 0
    n_tried   = 0
    seed      = 0
    wall_start = time.time()

    while n_saved < N_DEMOS and seed < MAX_SEEDS:
        n_tried += 1
        success, obs_list, action_list, reward_list, cube_start = \
            run_one_episode(env, expert, seed)

        if success:
            save_demo(n_saved, seed, obs_list, action_list, reward_list, cube_start)
            n_saved += 1
            print(f"[{n_saved:3d}/{N_DEMOS}] seed={seed:3d}  OK    "
                  f"steps={len(obs_list):4d}  saved episode_{n_saved-1:04d}.npz")
        else:
            print(f"        seed={seed:3d}  FAIL  steps={len(obs_list):4d}  (discarded)")

        seed += 1

    wall_elapsed = time.time() - wall_start

    # ---- Summary ----
    print(f"\n--- Done ---")
    print(f"Saved demos : {n_saved}")
    print(f"Tried seeds : {n_tried}")
    print(f"Success rate: {100*n_saved/n_tried:.1f}%")
    print(f"Wall clock  : {wall_elapsed:.1f}s")
    print(f"Output dir  : {DEMOS_DIR}/")

    env.close()


if __name__ == "__main__":
    main()
