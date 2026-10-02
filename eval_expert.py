"""
Step 1 of Phase 4: measure the scripted expert's success rate.

Runs N episodes HEADLESS (no viewer) so it goes fast. Counts how often
the cube actually ends up at the drop target. We want 90%+ before we
trust this expert to generate training demos.
"""

import time
import numpy as np
from so101_env import SO101PickPlaceEnv
from scripted_controller import ScriptedPickPlaceController


def main():
    n_episodes = 30

    env    = SO101PickPlaceEnv(render_mode=None,    # headless = fast
                               max_steps=2500,
                               randomize_cube=True)
    expert = ScriptedPickPlaceController(env)

    success_count    = 0
    success_steps    = []
    failure_episodes = []

    wall_start = time.time()

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        expert.reset()

        total_reward = 0.0
        for t in range(env.max_steps):
            action = expert.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        if info["success"]:
            success_count += 1
            success_steps.append(t + 1)
            tag = "OK    "
        else:
            failure_episodes.append(ep)
            tag = "FAIL  "

        print(f"Episode {ep:3d}: {tag}steps={t+1:4d}  reward={total_reward:7.2f}")

    wall_elapsed = time.time() - wall_start

    # ---- Summary ----
    print(f"\n--- Summary over {n_episodes} episodes ---")
    print(f"Success rate    : {success_count}/{n_episodes}  = {100*success_count/n_episodes:.1f}%")
    if success_steps:
        avg = np.mean(success_steps)
        print(f"Avg steps/succ  : {avg:.0f}  (= {avg*0.005:.1f}s sim time)")
        print(f"Min / Max steps : {min(success_steps)} / {max(success_steps)}")
    if failure_episodes:
        print(f"Failed seeds    : {failure_episodes}")
    print(f"Wall clock      : {wall_elapsed:.1f}s")
    print(f"Sim/wall speedup: {sum(success_steps + [2500]*len(failure_episodes)) * 0.005 / wall_elapsed:.1f}x")

    env.close()


if __name__ == "__main__":
    main()
