"""
Make a copy of a LeRobot dataset with the gripper ACTION replaced by two values.

Why: the leader arm's gripper reading drifted between recording sessions, so
the same intent ("close on the ball") was recorded as commands from ~59 down
to ~21. Low commands drive the follower's jaws hard into the ball or the
mechanical stop (servo overload), and the labels contradict each other.

The label is derived from the FOLLOWER's measured jaw (observation.state),
which does not drift. The measured jaw has three states:
    ~45        closed on nothing (home, approach)
    ~54-62     holding the ball (the ball stops the jaws)
    >= ~64     open
"Closed on nothing" and "holding" both mean "close the gripper", so:

    measured jaw `--lookahead` frames later <  --threshold  ->  --closed
    otherwise                                                ->  --open

The lookahead makes the label lead the jaw, since a command comes before the
jaw reaches it. Only the gripper dimension of `action` changes; states,
videos, and other columns are copied unchanged. Per-episode action stats in
meta/episodes_stats.jsonl are recomputed (LeRobot v2.1 aggregates them at
load time). The source dataset is never modified.

Usage (lerobot conda env):
    python relabel_gripper.py --src so101_ball_v2 --dst so101_ball_v4
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LOCAL_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "local"
GRIPPER = 5


def action_stats(actions: np.ndarray, count: int) -> dict:
    return {
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "count": [count],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="so101_ball_v2")
    p.add_argument("--dst", default="so101_ball_v4")
    p.add_argument("--closed", type=float, default=45.0)
    p.add_argument("--open", type=float, default=75.0)
    p.add_argument("--threshold", type=float, default=62.0, help="measured jaw below this = closed/holding")
    p.add_argument("--lookahead", type=int, default=5, help="frames the label leads the measured jaw")
    args = p.parse_args()

    src, dst = LOCAL_ROOT / args.src, LOCAL_ROOT / args.dst
    if not src.exists():
        raise SystemExit(f"Source dataset not found: {src}")
    if dst.exists():
        raise SystemExit(f"Destination already exists, refusing to overwrite: {dst}")

    print(f"Copying {src} -> {dst} (videos included, may take a minute)...")
    shutil.copytree(src, dst)

    parquet_files = sorted((dst / "data").rglob("episode_*.parquet"))
    new_stats = {}
    before_all, after_all, jaw_all = [], [], []

    for f in parquet_files:
        table = pq.read_table(f)
        col_type = table.schema.field("action").type
        actions = np.array(table.column("action").to_pylist(), dtype=np.float32)
        jaw = np.array(table.column("observation.state").to_pylist(), dtype=np.float32)[:, GRIPPER]

        future = np.concatenate([jaw[args.lookahead:], np.repeat(jaw[-1], args.lookahead)])
        before_all.append(actions[:, GRIPPER].copy())
        actions[:, GRIPPER] = np.where(future < args.threshold, args.closed, args.open)
        after_all.append(actions[:, GRIPPER].copy())
        jaw_all.append(jaw)

        new_col = pa.array(actions.tolist(), type=col_type)
        table = table.set_column(table.schema.get_field_index("action"), "action", new_col)
        pq.write_table(table, f)

        ep_idx = int(table.column("episode_index")[0].as_py())
        new_stats[ep_idx] = action_stats(actions.astype(np.float64), len(actions))

    stats_path = dst / "meta" / "episodes_stats.jsonl"
    out = []
    for line in stats_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rec["stats"]["action"] = new_stats[rec["episode_index"]]
        out.append(json.dumps(rec))
    if len(out) != len(parquet_files):
        raise SystemExit(f"Stats/episode mismatch: {len(out)} stats lines vs {len(parquet_files)} parquet files")
    stats_path.write_text("\n".join(out) + "\n", encoding="utf-8")

    before, after, jaw = map(np.concatenate, (before_all, after_all, jaw_all))
    holding = (jaw >= 50) & (jaw < args.threshold)
    print(f"\nEpisodes relabeled: {len(parquet_files)}   frames: {len(before)}")
    print(f"Gripper command BEFORE: min {before.min():.1f}  max {before.max():.1f}")
    print(f"Gripper command AFTER : closed={args.closed} on {(after == args.closed).mean():.0%}, "
          f"open={args.open} on {(after == args.open).mean():.0%}")
    print(f"Frames with jaw on the ball (50-{args.threshold:.0f}) labeled closed: "
          f"{(after[holding] == args.closed).mean():.0%}")
    print(f"\nNew dataset: local/{args.dst}")


if __name__ == "__main__":
    main()
