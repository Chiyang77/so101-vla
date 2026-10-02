"""
Quick tool to inspect a saved demo .npz file.

Usage:
    python inspect_demo.py demos/episode_0000.npz
    python inspect_demo.py demos/episode_0042.npz
"""

import sys
import numpy as np


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "demos/episode_0000.npz"

    print(f"Loading: {fname}\n")
    d = np.load(fname)

    print(f"Arrays in the file:")
    for key in d.keys():
        arr = d[key]
        print(f"  {key:15} shape={str(arr.shape):15} dtype={str(arr.dtype):10}")

    print(f"\nMetadata:")
    print(f"  seed       : {d['seed']}")
    print(f"  steps      : {d['observations'].shape[0]}")
    print(f"  duration   : {d['observations'].shape[0] * 0.005:.2f} s sim time")
    print(f"  cube_start : {d['cube_start']}")

    obs = d["observations"]
    print(f"\nObservation breakdown (18 dims):")
    print(f"  obs[0:6]   joint angles    first step: {obs[0, 0:6].round(3)}")
    print(f"                              last step: {obs[-1, 0:6].round(3)}")
    print(f"  obs[6:12]  joint velocities first step: {obs[0, 6:12].round(3)}")
    print(f"                              last step: {obs[-1, 6:12].round(3)}")
    print(f"  obs[12:15] cube position    first step: {obs[0, 12:15].round(3)}")
    print(f"                              last step: {obs[-1, 12:15].round(3)}")
    print(f"  obs[15:18] EE position      first step: {obs[0, 15:18].round(3)}")
    print(f"                              last step: {obs[-1, 15:18].round(3)}")

    acts = d["actions"]
    print(f"\nActions:")
    print(f"  first action: {acts[0].round(3)}")
    print(f"  last action : {acts[-1].round(3)}")

    rewards = d["rewards"]
    print(f"\nRewards:")
    print(f"  total       : {rewards.sum():.2f}")
    print(f"  per-step avg: {rewards.mean():+.3f}")
    print(f"  min / max   : {rewards.min():+.3f} / {rewards.max():+.3f}")


if __name__ == "__main__":
    main()
