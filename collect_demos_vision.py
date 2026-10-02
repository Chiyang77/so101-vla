"""
Collect VISION demos: run the scripted expert and record, at the DECIMATED
(50 Hz) control rate, the external-camera image + proprioception + action.

The expert still uses privileged state (via `info`) to run its IK — it's the
demo generator. But what we SAVE for the policy is only what a real robot could
see: the camera image + joint proprioception. The cube and green target are
visible in the image (nothing privileged is saved).

Each saved episode (demos_vision/episode_NNNN.npz):
  images  : (T, H, W, 3) uint8   external camera frames (50 Hz)
  proprio : (T, 12)      float32 [qpos[:6], qvel[:6]]
  actions : (T, 6)       float32 expert joint targets at each 50 Hz step
  target  : (2,)         float32 drop target (reference only; not used in training)
  seed    : ()           int32
"""

import os, time
import numpy as np
from so101_env import SO101PickPlaceEnv
from scripted_controller import ScriptedPickPlaceController

N_DEMOS   = 200
DECIM     = 4              # 200 Hz / 4 = 50 Hz control (matches state-based ACT)
IMG_SIZE  = 128
DEMOS_DIR = "demos_vision"
MAX_SEEDS = 300


def run_episode(env, expert, seed):
    obs, info = env.reset(seed=seed)
    expert.reset()
    target = env.drop_target_xy.copy()
    imgs_e, imgs_w, props, acts = [], [], [], []
    for t in range(env.max_steps):
        action = expert.act(obs, info)
        if t % DECIM == 0:                                   # record at 50 Hz
            imgs_e.append(env._render_external())           # 3/4 workspace view
            imgs_w.append(env._render_wrist())              # gripper close-up
            props.append(np.concatenate([env.data.qpos[:6],
                                         env.data.qvel[:6]]).astype(np.float32))
            acts.append(action.copy())
        obs, r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info["success"], imgs_e, imgs_w, props, acts, target


def main():
    os.makedirs(DEMOS_DIR, exist_ok=True)
    env = SO101PickPlaceEnv(obs_mode="state", img_size=IMG_SIZE,
                            max_steps=2500, randomize_cube=True,
                            goal_conditioned=True)
    expert = ScriptedPickPlaceController(env)

    saved, seed, t0 = 0, 0, time.time()
    while saved < N_DEMOS and seed < MAX_SEEDS:
        ok, imgs_e, imgs_w, props, acts, target = run_episode(env, expert, seed)
        if ok:
            np.savez_compressed(
                os.path.join(DEMOS_DIR, f"episode_{saved:04d}.npz"),
                images_ext=np.asarray(imgs_e, dtype=np.uint8),
                images_wrist=np.asarray(imgs_w, dtype=np.uint8),
                proprio=np.asarray(props, dtype=np.float32),
                actions=np.asarray(acts, dtype=np.float32),
                target=target.astype(np.float32),
                seed=np.int32(seed),
            )
            saved += 1
            if saved % 10 == 0 or saved == 1:
                print(f"[{saved:3d}/{N_DEMOS}] seed={seed:3d} T={len(imgs_e)} "
                      f"({time.time()-t0:.0f}s)")
        seed += 1

    print(f"\nSaved {saved} vision demos to {DEMOS_DIR}/  "
          f"(tried {seed} seeds, {time.time()-t0:.0f}s)")
    env.close()


if __name__ == "__main__":
    main()
