# VLA + Robot Arm Learning Plan

Goal: get hands-on with physics-AI / VLA by simulating in MuJoCo first, then deploying to a real **SO-ARM101** (LeRobot-compatible). Sim-first lets us validate interest, build the data pipeline, and reduce hardware risk before purchase.

End state: a language-conditioned policy (e.g. ACT, Diffusion Policy, or a small VLA) that we trained in sim, transferred to a real SO-101, and can demo doing pick-and-place from natural-language commands.

## Stack we want exposure to
- **MuJoCo** (CPU sim, MJCF, viewer) — primary physics engine
- **MuJoCo MJX** (JAX, GPU-parallel) — scale up training
- **LeRobot** (Hugging Face) — datasets, ACT, Diffusion Policy, SmolVLA, SO-101 drivers
- **NVIDIA Isaac Lab + Newton + Warp** — stretch goal in phase 4

---

## Phase 1 — MuJoCo fundamentals (1–2 weeks) ← we are here
- Install `mujoco` Python bindings; confirm with a "hello world" script.
- Run the interactive viewer (`python -m mujoco.viewer`) on a hand-written MJCF.
- Work through the official DeepMind MuJoCo tutorial colabs (concepts: MJCF, `mjModel` vs `mjData`, joints, actuators, contacts, sensors).
- Build one model from scratch in MJCF: a 2-link arm with a torque-controlled gripper.

Exit criterion: I can write an MJCF from scratch, simulate it, render it, and read state out of `mjData` in Python.

## Phase 2 — Drive the SO-101 in sim (1 week)
- Clone `mujoco_menagerie`, load the `trs_so_arm100` scene.
- Drive joints with position actuators in the viewer.
- Add a tabletop with a cube; tune contact/friction.
- Implement scripted IK (use `mink` or damped-least-squares) to reach a target pose.

Exit criterion: I can teleport the SO-101 end effector to any reachable point on the table.

## Phase 3 — Scripted pick-and-place + Gym wrapper (1 week)
- Add a camera body to the MJCF; render RGB observations.
- Wrap as a Gymnasium env (`reset`, `step`, observation = image + joint state, action = joint targets).
- Hand-script a pick-and-place policy that succeeds reliably.

Exit criterion: scripted policy hits 95%+ success on cube pick-and-place; can render demos to MP4.

## Phase 4 — Imitation learning (in progress)
Chose the **imitation/VLA branch** (direct on-ramp to hardware).

Progress so far (hand-rolled PyTorch, no LeRobot yet):
- ✅ Gym env (`so101_env.py`) + scripted expert controller (`scripted_controller.py`)
- ✅ Polar cube sampling → 94–98% expert success
- ✅ Collected 200 successful demos (`collect_demos.py` → `demos/`)
- ✅ Vanilla behavior cloning (`bc_*.py`) → **10%** success. Taught the BC
  failure modes: compounding error (distribution shift) + multimodality
  (state-machine phase not in the observation).
- ✅ Action chunking + temporal ensembling (`chunk_*.py`) → looked like 47%,
  but the success metric was too loose (only checked final cube XY).
- ⚠️ Tightened success to require a REAL lift (cube above 0.06 m at some point).
  Honest result: expert still 97%, but the **BC/chunk MLP = 0%** — its earlier
  "successes" were lucky nudges of cubes that spawned near the drop zone, not
  real grasps. Lesson: vanilla BC (even chunked+ensembled) can't do the
  contact-rich grasp here; metric design matters as much as the model.

- ✅ Hand-rolled ACT (`act_*.py`): transformer encoder-decoder + CVAE, GPU
  training. RTX 5080 (sm_120) works with torch **cu130** already in `warp` env
  (the earlier cu126 failure is gone) — train on GPU, ~20-90s/epoch.
- ⚠️ ACT @ 200 Hz = 0%. Root cause = the COPYCAT TRAP: cloning absolute joint
  targets at 200 Hz where each step's target is ~2-6% of joint travel from the
  current pose (median ~0) → policy learns "predict current pose" → arm stalls.
  Confirmed by data + research (ACT paper runs 50 Hz, LeRobot records 30 Hz).
- ✅ FIX: decimate control to 50 Hz (decimation=4 in `act_dataset.py`; hold each
  action 4 physics steps at inference in `eval_act.py`). **0% → 13.3%** — first
  real learned pick-and-place successes.
- ✅ Decomposed the task (`task="pick_lift"` mode in env): the GRASP is solved —
  **86.7%** pick-and-lift. Full pick-place was low because PLACING was weak, and
  placing was weak because the drop target was NOT in the observation.
- ✅ Goal-conditioning: randomized drop target each episode, appended to obs
  (18->20), green marker in scene, expert places at it. `goal_conditioned=True`.
- ✅ Fixed expert place bugs (found via watching + validating demos):
  (1) old PLACE used held-cube height -> placed in air; (2) one-step-lag let
  states skip; (3) RELEASE rose while gripping -> flung cube; (4) **LIFT used
  cube_z+offset -> runaway target that fully extended the arm** (the user's "why
  straighten the arm?" observation). Fix = absolute LIFT_Z + hover/lower/release.
  Expert: 75% -> **99%**, placement 3.7cm -> **0.75cm**, cube stays low (~0.09m).
- ✅ Demo validation (`validate_demos.py`) from RECORDED obs (NOT open-loop replay,
  which diverges through contacts). 200/200 demos clean. Lesson: validate data
  before training; open-loop replay of closed-loop demos is not a valid check.
- ✅ Goal-conditioned ACT on 200 validated clean demos: **83.3%** (25/30) on the
  strict metric with RANDOM targets. PHASE 4 EXIT CRITERION MET (~80%+).
  Full arc: vanilla BC 0% -> ACT@200Hz 0% (copycat) -> ACT@50Hz fixed-target 13%
  -> ACT@50Hz goal-conditioned+clean-demos **83.3%**.
  Remaining failures undershoot placement on the hardest target configs.

- ✅ Found the place was INCOMPLETE: success fired while the cube was still
  GRIPPED (held at z<0.05), so demos stopped before the release and the policy
  never learned to open the gripper. Fix = success requires the cube SETTLED
  (z<0.03, only true after release). Re-collected (demos now include the
  gripper-open), retrained: **86.7%** on the complete place-AND-release task.
  Lesson: a success criterion satisfiable mid-action teaches an incomplete skill.

Final Phase 4 result: **86.7%** complete pick-place-release, random targets, strict metric.
Levers to push further: more demos (500), bigger chunk (K=100), revive CVAE (beta
0.1-1; KL collapses), add wrist camera (vision). Or proceed to Phase 5 (buy SO-101).

Key files for eval: `eval_act.py`, `eval_chunk.py`, `eval_bc.py`.
Use the `warp` conda env python (`%USERPROFILE%\anaconda3\envs\warp\python.exe`)
for all ML scripts — GPU works there now.

Next levers to push past 13%: revive the CVAE latent (KL collapsed to ~0 with
beta=10 → try beta 0.1-1), bigger chunk (K=100 like ACT paper), more demos, add
the wrist camera for vision. Open-source refs: EE5108-DigitalTwins/
lerobot_mujoco_sim (MuJoCo+SO101+ACT), StoneT2000/lerobot-sim2real (PPO sim2real).

Exit criterion: a learned policy that reliably (~80%+) picks and places in sim.

## Phase 4b — Vision policy (ACT from pixels) — IN PROGRESS
Prerequisite for real deployment: the state policy uses privileged sim
coordinates (cube_pos, target_xy) the real robot can't provide. So swap the
observation to a CAMERA IMAGE + proprioception.
- ✅ Added external camera to scene (`scene_table.xml`, fovi'd 3/4 view; sees
  arm + red cube + green target). Env `obs_mode="pixels"` returns
  {image 128x128x3 uint8, proprio [qpos[:6],qvel[:6]]}. No privileged state.
- ✅ Vision pipeline (`*_vision.py`): CNN image encoder (128px -> 16 tokens) ->
  ACT transformer + CVAE + chunking. Random-shift image aug. 8.4M params.
- ✅ Collected 200 vision demos (`collect_demos_vision.py` -> `demos_vision/`,
  463 MB, images recorded at 50 Hz decimated rate).
- ✅ Trained vision ACT (val_L1 0.077, even lower than state). Eval: **16.7%**
  (5/30) strict, random targets. PIPELINE WORKS (learns pick-place from PIXELS +
  proprio, zero privileged state) but precision is weak: failures show the cube
  dragged INWARD (x~0.16-0.20 vs 0.22-0.27 spawn) — the policy sees & reaches the
  cube but the grasp isn't precise enough from one low-res external view.
  Loss-vs-success gap again (low val loss, low closed-loop success).
- ✅ DUAL-CAMERA (external + wrist) vision ACT: **80.0%** (24/30)! Added the
  wrist camera as a 2nd view (`*_vision.py` now take {image_ext, image_wrist,
  proprio}; separate CNN encoder per view + per-camera embedding, 9.4M params).
  The wrist cam supplies the close-up grasp view the external lacked -> grasp
  precision fixed. val_L1 0.051. Vision pick-place-release from PURE PIXELS
  (no privileged coords), approaching the state-based 86.7%.

**Phase 4b DONE.** This is the sim-to-real-ready policy: uses only what the real
robot has (2 cameras + joint encoders), and the two sim cameras (external + wrist)
map exactly onto the real Hiwonder kit's two cameras.

Full vision arc: single-cam external 16.7% -> dual-cam (external+wrist) 80.0%.
Lesson: diagnose the failure MODE (reaches cube, fumbles grasp = precision, not
perception) -> the fix (wrist close-up view) was targeted, not a guess.

Levers to push past 80%: pretrained ResNet encoder, more demos, higher-res wrist,
color-jitter aug (also helps sim-to-real).

## Phase 5 — Real SO-101 (1–2 weeks)
- HARDWARE: **Hiwonder SO-ARM101 Advanced Kit, Assembled**. Leader+follower,
  dual cameras (wrist 1080p + external 480p), HX-10HM/HX-30HM 30kg 12V servos
  (register layout claimed identical to Feetech STS3215), BusLinker V3.0 board.
  It IS the SO-101 geometry (PLA) so our MuJoCo model / policy transfer.
- PREREQUISITE before real deploy: our policy uses privileged sim state
  (exact cube/target pos). Real robot can't provide that -> must first train a
  VISION-based policy (wrist camera image -> action) in sim. That's the true
  next technical step and needs no hardware.
- First check: confirm Hiwonder's LeRobot snapshot actually drives the arm.
- Then: assemble/calibrate, collect ~30 real teleop demos with the leader arm,
  fine-tune / co-train the sim policy on real data.

Exit criterion: real SO-101 picks up a cube on command. Demo video.

### Phase 5 results (real-robot ACT, red Hol-ee ball -> fenced green circle)
| Eval | Model / change | Result |
|---|---|---|
| v4 | 30 demos, cluttered room | reaches near ball, misses |
| v5 | 50 demos (+20 recovery) + temporal ensembling | 3/5 |
| v8 | 70 demos, midday light (train was night) | 3/8; +fence on circle 6/9 |
| v10 | 96 demos, dusk | 2/10 (leader gripper drift found) |
| v6_snap | relabel bug: "holding ball" labeled open | never closes |
| **eval_v7** | **96 demos, gripper relabeled from measured jaw (45/75), snap, night** | **7/10 clean (+1 near-complete)** |

Lessons: lighting must match training (day vs night); pauses in demos force long
action horizons; recovery demos fix "stuck near ball"; a rim on the target stops
roll-off; leader gripper drifted across sessions -> label the gripper from the
follower's MEASURED jaw (3 states: ~45 empty, ~58 holding, 64+ open); with binary
labels, snap the gripper output at inference. Tools: go_home.py, run_eval.bat,
relabel_gripper.py, GRIPPER_SNAP env var in lerobot record.py.

## Phase 6 (stretch) — Isaac Lab / Newton / VLA
- Port the same scene to Isaac Lab (Newton backend).
- Fine-tune SmolVLA or OpenVLA on our dataset for language conditioning ("pick the red block").

---

## Key resources
- `mujoco_menagerie` — robot model zoo, includes SO-ARM100
- DeepMind MuJoCo tutorials (the official colabs)
- `mink` — clean Python IK for MuJoCo
- LeRobot repo — Gym envs + ACT/Diffusion Policy/SmolVLA
- MJX docs — for phase 4 RL branch

## Notable risks / decisions
- MJX needs JAX, which is rough on Windows native. Plan to set up WSL2 (Ubuntu) before phase 4.
- Phase 4 branch choice is the key fork. Default: imitation branch, since it's the direct on-ramp to real-arm work.
- Hardware purchase deferred until end of phase 4 — once we have a checkpoint that's worth deploying.
