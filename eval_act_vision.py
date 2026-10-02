"""
Evaluate the vision ACT on fresh seeds (1000+), strict metric.

Runs the env in STATE mode (fast, no per-step render) and renders the external
camera only at the decimated (50 Hz) rate to feed the policy — mirroring how
demos were collected. Each policy action is held for `decim` physics steps.

Usage:
    python eval_act_vision.py                # headless, 30 episodes
    python eval_act_vision.py --no-ensemble
"""

import sys
import numpy as np
from so101_env import SO101PickPlaceEnv
from act_policy_vision import VisionACTController

DECIM = 4


def proprio_of(env):
    return np.concatenate([env.data.qpos[:6], env.data.qvel[:6]]).astype(np.float32)


def main():
    ensemble = "--no-ensemble" not in sys.argv
    n_episodes = 30

    env = SO101PickPlaceEnv(obs_mode="state", img_size=128, max_steps=2500,
                            randomize_cube=True, goal_conditioned=True,
                            task="pick_place")
    policy = VisionACTController(ensemble=ensemble)
    print(f"chunk={policy.K} ensemble={ensemble} decim={DECIM} device={policy.device}\n")

    success = 0
    for ep in range(n_episodes):
        seed = 1000 + ep
        _, info = env.reset(seed=seed)
        policy.reset()
        t, done = 0, False
        while t < env.max_steps and not done:
            vobs = {"image_ext": env._render_external(),
                    "image_wrist": env._render_wrist(),
                    "proprio": proprio_of(env)}
            action = policy.act(vobs, info)
            for _ in range(DECIM):
                _, r, terminated, truncated, info = env.step(action)
                t += 1
                if terminated or truncated:
                    done = True
                    break
        ok = info["success"]; success += int(ok)
        print(f"Episode {ep:3d} (seed {seed}): {'OK  ' if ok else 'FAIL'}  "
              f"steps={t:4d}  cube_end={info['cube_pos'][:2].round(3)}")

    print(f"\n--- Vision ACT success: {success}/{n_episodes} "
          f"= {100*success/n_episodes:.1f}% ---")
    env.close()


if __name__ == "__main__":
    main()
