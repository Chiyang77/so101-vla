"""
Evaluate the trained ACT on fresh seeds (1000+), strict success metric
(cube must be genuinely lifted). Same protocol as eval_chunk.py.

Usage:
    python eval_act.py                 # headless, 30 episodes, ensembling on
    python eval_act.py --render
    python eval_act.py --no-ensemble
"""

import sys
import numpy as np
from so101_env import SO101PickPlaceEnv
from act_policy import ACTController


def main():
    render   = "--render" in sys.argv
    ensemble = "--no-ensemble" not in sys.argv
    lookahead = 0
    if "--lookahead" in sys.argv:
        lookahead = int(sys.argv[sys.argv.index("--lookahead") + 1])
    task = "pick_lift" if "--pick-lift" in sys.argv else "pick_place"

    n_episodes = 30
    env = SO101PickPlaceEnv(render_mode="human" if render else None,
                            max_steps=2500, randomize_cube=True, task=task,
                            goal_conditioned=True)
    print(f"task={task}")
    policy = ACTController(ensemble=ensemble, lookahead=lookahead)
    print(f"chunk_size={policy.K}  ensemble={ensemble}  lookahead={lookahead}\n")

    # Run the policy at the DECIMATED rate: query once, then hold the action
    # for `decim` physics steps. This matches how the policy was trained
    # (50 Hz control, not 200 Hz).
    decim = policy.decimation
    print(f"control decimation={decim} ({200//decim} Hz policy)\n")

    success = 0
    for ep in range(n_episodes):
        seed = 1000 + ep
        obs, info = env.reset(seed=seed)
        policy.reset()
        t = 0
        done = False
        while t < env.max_steps and not done:
            action = policy.act(obs, info)        # one decimated-rate decision
            for _ in range(decim):                # hold it for `decim` steps
                obs, reward, terminated, truncated, info = env.step(action)
                t += 1
                if render:
                    env.render()
                if terminated or truncated:
                    done = True
                    break
        ok = info["success"]
        success += int(ok)
        # In render mode, keep showing a moment after success so the cube fully
        # lands on the target (success fires mid-fall; this lets you SEE it settle).
        if render and ok:
            last = policy.act(obs, info)
            for _ in range(40):
                obs, *_ , info = env.step(last)
                env.render()
        print(f"Episode {ep:3d} (seed {seed}): {'OK  ' if ok else 'FAIL'}  "
              f"steps={t+1:4d}  cube_end={info['cube_pos'][:2].round(3)}")

    print(f"\n--- ACT success rate: {success}/{n_episodes} "
          f"= {100*success/n_episodes:.1f}% ---")
    env.close()


if __name__ == "__main__":
    main()
