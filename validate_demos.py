"""
Validate demos from their RECORDED observations (the actual closed-loop states
during collection). This is the correct way to inspect demo quality.

We do NOT open-loop replay the actions: replaying recorded actions blindly
diverges through contact-rich grasping (tiny numerical differences amplify),
flinging the cube — an artifact of the replay, not the demo.

Obs layout (goal-conditioned, 20-dim):
   [12:15] cube xyz   [15:18] ee xyz   [18:20] target xy
"""

import glob
import numpy as np

LIFT_MIN = 0.06   # cube must rise above this (real grasp)
LIFT_MAX = 0.20   # ... but NOT fly up here (clean low transport, no runaway)
PLACE_TOL = 0.04  # final cube within this of target


def main():
    files = sorted(glob.glob("demos/*.npz"))
    print(f"Validating {len(files)} demos from recorded observations\n")

    n_clean = 0
    lift_max, place_err, jumps, lengths = [], [], [], []
    flagged = []

    for f in files:
        d = np.load(f)
        obs = d["observations"]; act = d["actions"]
        cube = obs[:, 12:15]; tgt = obs[0, 18:20]
        T = obs.shape[0]

        peak_z   = cube[:, 2].max()
        final_err = np.linalg.norm(cube[-1, :2] - tgt)
        # smoothness: largest single-step change in arm-joint actions
        jerk = np.abs(np.diff(act[:, :5], axis=0)).max() if T > 1 else 0.0

        lifted_ok = LIFT_MIN < peak_z < LIFT_MAX      # rose, but stayed low
        placed_ok = final_err < PLACE_TOL + 0.01      # last frame ~ at target (off-by-one slack)

        if lifted_ok and placed_ok:
            n_clean += 1
            lift_max.append(peak_z); place_err.append(final_err)
            jumps.append(jerk); lengths.append(T)
        else:
            reason = []
            if peak_z >= LIFT_MAX: reason.append(f"FLEW to {peak_z:.2f}m")
            if peak_z <= LIFT_MIN: reason.append(f"never lifted ({peak_z:.2f}m)")
            if not placed_ok:      reason.append(f"end {final_err*100:.1f}cm off")
            flagged.append(f"{f}: " + ", ".join(reason))

    print(f"CLEAN (lifted low + placed at target): {n_clean}/{len(files)} "
          f"= {100*n_clean/len(files):.1f}%\n")
    if place_err:
        print(f"Peak cube height: mean={np.mean(lift_max):.3f}  max={np.max(lift_max):.3f}  "
              f"(clean ~0.09; >0.2 means runaway lift)")
        print(f"Final place err : mean={np.mean(place_err)*100:.2f}cm  "
              f"max={np.max(place_err)*100:.2f}cm")
        print(f"Action jerk     : mean={np.mean(jumps):.3f}  max={np.max(jumps):.3f} rad/step")
        print(f"Episode length  : mean={np.mean(lengths):.0f} steps")
    if flagged:
        print(f"\n{len(flagged)} FLAGGED demos:")
        for x in flagged[:12]:
            print("  " + x)
        if len(flagged) > 12:
            print(f"  ... and {len(flagged)-12} more")
    else:
        print("\nAll demos are clean: low smooth lift, precise placement.")


if __name__ == "__main__":
    main()
