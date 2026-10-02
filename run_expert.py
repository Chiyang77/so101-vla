"""
Run the scripted expert controller inside the Gym env.

This is the "proof" that the env + controller decoupling works:
the env doesn't know it's being driven by a scripted policy, and
the controller doesn't know it's running inside an env wrapper.
Same interface a learned policy will use in Phase 4.
"""

import numpy as np
from so101_env import SO101PickPlaceEnv
from scripted_controller import ScriptedPickPlaceController


def main():
    env = SO101PickPlaceEnv(render_mode="human",
                            max_steps=2500,
                            randomize_cube=True,
                            goal_conditioned=True)   # random target, shown as green disk
    expert = ScriptedPickPlaceController(env)

    n_episodes = 5
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        expert.reset()

        total_reward = 0.0
        for t in range(env.max_steps):
            action = expert.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            total_reward += reward
            if terminated or truncated:
                break

        print(f"Episode {ep}: success={info['success']}, "
              f"total_reward={total_reward:7.2f}, steps={t+1}")

    env.close()


if __name__ == "__main__":
    main()
