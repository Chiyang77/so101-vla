"""
Drive the SO-101 follower back to the demo start ("home") pose.

Policy-driven evals have no teleop, so after a failed episode the next one
starts wherever the arm was left. Run this between eval episodes so every
episode starts from the pose the demos started from.

The arm moves along a smoothed joint-space path over --duration seconds. By
default torque stays ON when the script exits, so the arm holds the home pose
until the record script connects. Ctrl+C stops the motion in place.

Usage (lerobot conda env):
    python go_home.py
    python go_home.py --duration 6          # slower
    python go_home.py --release             # go limp at the end instead of holding
    python go_home.py --dry-run             # print current vs home, don't move
"""

import argparse
import time

import numpy as np

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Mean first-frame observation.state of so101_ball_v2 episodes 0-49
# (normalized units: body joints -100..100, gripper 0..100).
HOME = np.array([2.15, -98.74, 99.48, 5.78, -53.32, 45.46])

CONTROL_HZ = 50


def read_pose(robot):
    obs = robot.get_observation()
    return np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=float)


def send_pose(robot, pose):
    robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, pose)})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="COM24")
    p.add_argument("--id", default="my_awesome_follower_arm")
    p.add_argument("--duration", type=float, default=4.0, help="seconds for the move")
    p.add_argument("--release", action="store_true", help="disable torque on exit (arm goes limp)")
    p.add_argument("--dry-run", action="store_true", help="report poses without moving")
    args = p.parse_args()

    cfg = SO101FollowerConfig(port=args.port, id=args.id, disable_torque_on_disconnect=args.release)
    robot = SO101Follower(cfg)
    robot.connect()  # may ask to press ENTER to load the saved calibration

    try:
        start = read_pose(robot)
        err = np.abs(start - HOME)
        print("joint           current     home    |diff|")
        for j, c, h, e in zip(JOINTS, start, HOME, err):
            print(f"{j:14s} {c:8.1f} {h:8.1f} {e:8.1f}")

        if args.dry_run:
            return
        if err.max() < 2.0:
            print("Already home.")
            return

        steps = max(1, int(args.duration * CONTROL_HZ))
        for i in range(1, steps + 1):
            t = i / steps
            s = t * t * (3 - 2 * t)  # smoothstep: zero velocity at both ends
            send_pose(robot, start + s * (HOME - start))
            time.sleep(1.0 / CONTROL_HZ)

        time.sleep(0.5)
        final_err = np.abs(read_pose(robot) - HOME)
        print(f"Done. Max joint error from home: {final_err.max():.1f} "
              f"({JOINTS[int(final_err.argmax())]})")
    except KeyboardInterrupt:
        print("\nStopped by user; arm holds its current position.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
