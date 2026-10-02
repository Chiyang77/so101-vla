"""
Smoke-test the SO101PickPlaceEnv with random actions.

Goal: confirm the Gymnasium interface works — reset, step, spaces, etc.
We don't expect the cube to actually be picked up by random actions;
this is purely about validating the contract.
"""

import numpy as np
from so101_env import SO101PickPlaceEnv


def main():
    # ---- 1. Construct env ----
    env = SO101PickPlaceEnv(render_mode="human", max_steps=500)

    print(f"Action space      : {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Action low        : {env.action_space.low}")
    print(f"Action high       : {env.action_space.high}")
    print(f"Observation space : {env.observation_space}")
    print(f"Obs shape         : {env.observation_space.shape}\n")

    # ---- 2. Run a few episodes ----
    for episode in range(3):
        obs, info = env.reset(seed=episode)
        print(f"--- Episode {episode} ---")
        print(f"  Initial cube pos : {np.round(info['cube_pos'], 3)}")
        print(f"  Initial EE pos   : {np.round(info['ee_pos'], 3)}")
        print(f"  obs shape        : {obs.shape}, dtype: {obs.dtype}")

        ep_reward = 0.0
        for t in range(200):
            # Random action sampled inside the legal range
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            ep_reward += reward

            if terminated or truncated:
                break

        print(f"  Episode reward   : {ep_reward:.2f}")
        print(f"  Steps            : {info['step']}")
        print(f"  Cube end pos     : {np.round(info['cube_pos'], 3)}")
        print(f"  Success          : {info['success']}\n")

    env.close()
    print("Env smoke test complete.")


if __name__ == "__main__":
    main()
