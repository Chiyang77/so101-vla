"""
Diagnostic tool: re-run a specific seed with the viewer open and detailed
per-state logging. Useful for seeing exactly what went wrong on a failed
seed (since failed demos aren't saved to disk).

Usage:
    python replay_seed.py             # default seed = 2 (a known failure)
    python replay_seed.py 11          # specific seed
    python replay_seed.py 11 1000     # specific seed + max steps
"""

import sys
import numpy as np
from so101_env import SO101PickPlaceEnv
from scripted_controller import ScriptedPickPlaceController


def main():
    # Parse command-line arguments (with defaults)
    seed       = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    max_steps  = int(sys.argv[2]) if len(sys.argv) > 2 else 2500

    print(f"Replaying seed={seed}, max_steps={max_steps}\n")

    env    = SO101PickPlaceEnv(render_mode="human",
                               max_steps=max_steps,
                               randomize_cube=True)
    expert = ScriptedPickPlaceController(env)

    obs, info = env.reset(seed=seed)
    expert.reset()

    print(f"Initial cube position: {info['cube_pos'].round(3)}")
    print(f"Drop target         : {env.drop_target_xy}\n")
    print(f"{'step':>5}  {'state':<12}  {'EE pos':<24}  {'cube pos':<24}  "
          f"{'dist_ee_cube':>12}  reward")
    print("-" * 100)

    last_state = ""
    total_reward = 0.0

    for t in range(max_steps):
        action = expert.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward

        current_state = expert.STATES[expert.state_idx]

        # Print on state transition + every 100 steps
        if current_state != last_state or t % 100 == 0:
            ee  = info["ee_pos"]
            cube = info["cube_pos"]
            d   = np.linalg.norm(ee - cube)
            tag = ">>>" if current_state != last_state else "   "
            print(f"{tag}{t:5d}  {current_state:<12}  "
                  f"{str(ee.round(3)):<24}  {str(cube.round(3)):<24}  "
                  f"{d:>12.4f}  {reward:+.3f}")
            last_state = current_state

        if terminated or truncated:
            break

    print(f"\n--- Result ---")
    print(f"Final state    : {expert.STATES[expert.state_idx]}")
    print(f"Total steps    : {t+1}")
    print(f"Total reward   : {total_reward:.2f}")
    print(f"Terminated     : {terminated} (success)")
    print(f"Truncated      : {truncated} (timeout)")
    print(f"Final cube pos : {info['cube_pos'].round(3)}")
    print(f"Final EE pos   : {info['ee_pos'].round(3)}")
    print(f"Success        : {info['success']}")

    env.close()


if __name__ == "__main__":
    main()
