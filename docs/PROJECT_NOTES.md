# VLA Project — Deep Notes

A living, detailed reference for this project: what every file does, the
knowledge behind each piece, the full journey of what we built and learned, and
how to run things. Expanded as the project grows.

> Companion docs: [CVAE_explained.md](CVAE_explained.md) (deep dive on the CVAE).
> Quick reference: [../README.md](../README.md). Roadmap & status: [../plan.md](../plan.md).

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [Environment & tooling](#2-environment--tooling)
3. [The big picture (pipeline)](#3-the-big-picture-pipeline)
4. [Concepts / the knowledge behind it](#4-concepts--the-knowledge-behind-it)
5. [File-by-file reference](#5-file-by-file-reference)
6. [The journey: what we tried and learned](#6-the-journey-what-we-tried-and-learned)
7. [Key lessons](#7-key-lessons)
8. [How to run everything](#8-how-to-run-everything)
9. [Current status & next steps](#9-current-status--next-steps)
10. [Glossary](#10-glossary)

---

## 1. What this project is

A **sim-first** path to learning physics-AI / robot manipulation. We teach a
simulated **SO-ARM101** (a low-cost 6-DOF arm, LeRobot-compatible) to do
**pick-and-place** in **MuJoCo**, by:

1. building the environment + a scripted expert,
2. collecting demonstrations,
3. training a neural-network policy by **imitation learning** to reproduce them.

End goal: deploy to a real SO-101 (Phase 5) and progress toward
Vision-Language-Action (VLA) models. We hand-roll everything to understand it,
rather than calling a framework (LeRobot) as a black box.

**Current headline result:** a goal-conditioned **ACT** policy does a complete
pick-place-**release** to a random target with **~86.7%** success (strict metric).

---

## 2. Environment & tooling

- **OS / shell:** Windows 11, PowerShell.
- **Python env:** conda env **`warp`**. Use its interpreter for ALL scripts:
  `%USERPROFILE%\anaconda3\envs\warp\python.exe`
  It has: `torch` (CUDA 13 / cu130), `mujoco`, `mink`, `gymnasium`, `numpy`.
- **GPU:** RTX 5080 (Blackwell, compute capability sm_120). Works with the
  `torch 2.12.0+cu130` already in `warp`. (An earlier cu126 build did NOT support
  sm_120 and forced CPU — that's resolved.)
- **Always run from the project root** so relative paths (`mujoco_menagerie/...`,
  `demos/`) resolve.
- **Background runs:** long jobs (collect/train/eval) are run with `python -u`
  (unbuffered) and redirected to a `.log` so progress is visible live and never
  lost to buffering.

---

## 3. The big picture (pipeline)

```
                       so101_env.py        ← the world (SO-101 + table + cube + goal)
                            │
  scripted_controller.py    │  ← the EXPERT (IK + state machine): generates demos
            │               │
            └──► collect_demos.py ──► demos/*.npz   ← recorded (obs, action) trajectories
                            │
                  validate_demos.py     ← check demos are clean BEFORE training
                            │
                  act_dataset.py        ← demos → training pairs (50 Hz, normalized, chunked)
                            │
   act_model.py ──────► act_train.py ──► act_policy.pt + act_stats.npz   ← THE MODEL
   (network def)            │            (weights)       (normalization)
                            │
                  act_policy.py          ← wrap weights as a controller (z=0, ensembling)
                            │
                  eval_act.py            ← run / score / render the policy
```

**Two artifacts ARE the trained model:** `act_policy.pt` (network weights) and
`act_stats.npz` (obs/action normalization + chunk size + decimation). Inference
needs both.

**Three roles to keep straight:**
- **Environment** (`so101_env.py`) — physics + task. Passive: provides `reset`/`step`.
- **Controller** — the "brain" that outputs actions. Either the *scripted expert*
  (`scripted_controller.py`) or the *learned policy* (`act_policy.py`). Same
  `reset()`/`act(obs)` interface, so they're swappable.
- **Driver loop** — code that calls `reset`/`step` in a loop (`collect_demos.py`,
  `eval_act.py`). The env has NO loop of its own.

---

## 4. Concepts / the knowledge behind it

### 4.1 MuJoCo basics
- **MJCF**: the XML model format. `<body>` nesting = the kinematic tree;
  `<joint>` gives a body degrees of freedom; `<geom>` = shape (collision +
  visual); `<site>` = a named massless reference point; `<actuator>` = motor.
- **`mjModel`** = static description (parsed from XML, read-only). **`mjData`** =
  state (`qpos`, `qvel`, `ctrl`, `time`, contacts). **`mj_step(model, data)`**
  advances `data` one timestep. This triad is every MuJoCo program.
- **`qpos` vs Cartesian**: `qpos` are *generalized coordinates* (joint angles; a
  free object adds 7 = xyz + quaternion). World XYZ of bodies/sites
  (`data.site_xpos`) is *derived* from `qpos` via forward kinematics.
- **Units**: SI — metres, radians, seconds, kg, N.
- **Quaternion**: 4 numbers `(w,x,y,z)` encoding a 3D rotation (no gimbal lock,
  must be unit length). A free joint's pose = 3 position + 4 quaternion = 7 in
  `qpos`, but only 6 in `qvel` (3 linear + 3 angular).

### 4.2 The SO-101 model & inverse kinematics
- Model from **mujoco_menagerie** (`robotstudio_so101`). 6 actuated joints:
  `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
  Uses **`<position>` actuators** (you command a *target angle*; a built-in PD
  servo drives there) — unlike a `<motor>` (raw torque).
- **Forward kinematics (FK)**: joint angles → end-effector pose (easy; MuJoCo does
  it). **Inverse kinematics (IK)**: desired end-effector pose → joint angles
  (hard; many/no solutions). We use **`mink`** (a MuJoCo IK library): declare a
  `FrameTask` (drive the `gripperframe` site to a target) + a `PostureTask` (keep
  the wrist pointing down), call `solve_ik` each step. Closed-loop: it re-reads
  the true `qpos` every step and corrects.

### 4.3 The Gym environment contract
- `reset(seed) → (obs, info)` starts an episode; `step(action) →
  (obs, reward, terminated, truncated, info)` advances one timestep.
- **`terminated`** = task ended naturally (success/failure). **`truncated`** =
  cut off by a time limit. The distinction matters for RL value bootstrapping
  (not for us, since we do IL).
- `action_space` / `observation_space` declare shapes & bounds. We pull the
  action bounds straight from the model's `actuator_ctrlrange` — the XML is the
  single source of truth for joint limits.
- **The env is passive** — it owns physics but has no loop. The driver code loops.

### 4.4 The scripted expert (state machine + IK)
- A finite-state machine: `HOME → OPEN_GRIP → HOVER → DESCEND → GRASP → LIFT →
  PLACE_HOVER → PLACE_DOWN → RELEASE` (loops). Each state sets an IK target and a
  transition condition (`arrived(tol)` OR a timeout). IK turns the Cartesian
  target into the 6 joint commands each step.
- It is the **demo generator**, not part of the learned policy. It has privileged
  access (reads true state, runs IK); the learned policy gets only the obs.

### 4.5 Imitation learning vs reinforcement learning
- **We use Imitation Learning (IL)** — specifically **behavior cloning**: collect
  expert demos, train a network to map `obs → action` by supervised regression
  (L1/MSE). No reward, no exploration.
- **RL** (NOT used here) learns from a reward signal by trial and error. The env
  has a `_get_reward()` but it is **unused** — leftover scaffolding. The CVAE,
  the word "policy", and the Gym interface can *look* RL-ish but none of this is RL.
- Why IL: sample-efficient (hundreds of demos, not millions of episodes), no
  reward engineering, and it's how real-robot VLAs (ACT, Diffusion Policy, π0,
  SmolVLA) are trained. The expert→demos→clone pipeline is the standard recipe.

### 4.6 The two failure modes of naive behavior cloning
1. **Distribution shift / compounding error**: the policy only ever saw "good"
   states (the expert was good). One small error → an unfamiliar state → bigger
   error → spiral. The fix family: action chunking (fewer decision points),
   temporal ensembling, more data.
2. **Multimodality**: the same observation can have several valid actions; L1
   loss learns their **average**, which is often invalid (mushy grasp). The fix:
   the **CVAE** (see 4.9) — and putting enough info in the observation that the
   right action is less ambiguous (e.g. goal-conditioning).

### 4.7 Action chunking & temporal ensembling
- **Chunking**: predict the next **K** actions from one observation (not just 1).
  Fewer decision points → errors compound K× less, and a chunk commits to ONE
  coherent intent instead of averaging per-step. Core idea of ACT.
- **Temporal ensembling** (run-time): query the model every step; for the current
  timestep, average all overlapping chunk predictions made in recent steps
  (recency-weighted, `w = exp(-m·age)`). Cancels per-prediction jitter; this is
  what made the chunk policy smooth.

### 4.8 The CONTROL-RATE / copycat trap (the biggest single fix)
- We collect at MuJoCo's 200 Hz, but the expert's per-step target is only
  ~2–6 % of a joint's travel from the current pose (median ≈ 0). So "predict the
  current pose" is a near-zero-loss shortcut → the policy outputs ≈ "stay put" →
  the arm **stalls** and never reaches the cube. Low val loss, broken behavior.
- **Fix: decimate control to ~50 Hz** (`decimation=4` in `act_dataset.py`; at
  inference, hold each predicted action for 4 physics steps in `eval_act.py`).
  Now each step's motion is large enough that copycat is no longer near-optimal.
  This single change took us from **0% → 13%** (and unlocked everything after).
- Backed by literature: the ACT paper runs at **50 Hz**, LeRobot records SO-101 at
  **30 Hz**. Nobody clones at 200 Hz.

### 4.9 ACT architecture (transformer + CVAE)
- **ACT = Action Chunking Transformer.** Our minimal version (`act_model.py`) has
  three transformer pieces:
  1. **CVAE encoder** (training only): `[CLS, obs, expert_chunk] → (mu, logvar)`
     of a latent `z`.
  2. **Main encoder**: `[obs, z] → memory`.
  3. **Decoder**: K learned query tokens cross-attend to memory → K actions.
- **CVAE = Conditional Variational Auto-Encoder.** It cures multimodality: during
  training the encoder *peeks at the expert's action* and assigns conflicting-but-
  valid actions **different `z`** values, so the decoder maps `(obs, z) → action`
  with no collision (no averaging). At test, `z = 0` (prior mean) selects one
  coherent mode. The loss is `L1(chunk) + β·KL(z‖N(0,1))`; the **reparameterization
  trick** (`z = mu + σ·ε`) lets gradients flow through the random sample. Full
  detail in [CVAE_explained.md](CVAE_explained.md).
- **KL collapse in our runs**: with β=10, `train_KL → 0` — the latent went unused
  and ACT became a deterministic chunk predictor. It still hit 86.7% because our
  *state-based, goal-conditioned* task is barely multimodal (the obs nearly
  determines the action). The CVAE matters more with vision / human demos; then
  lower β (~0.1–1) or anneal it.

### 4.10 Goal-conditioning
- Originally the drop target was a fixed constant NOT in the observation — the
  policy had to *memorize* one location, and placed poorly. **Goal-conditioning**:
  randomize the target each episode and **append it to the observation**
  (`obs` 18 → 20). Now the policy is *told* where to place and generalizes to any
  target. This took the full task from ~13% (memorized) toward 80%+.

### 4.11 Demo validation & success-criterion design (two subtle, important ideas)
- **Validate data before training.** We checked the recorded demos for: real lift
  (cube rises but stays LOW, not flung), precise placement, smooth actions.
  Caveat learned the hard way: you CANNOT validate by *open-loop replaying* the
  recorded actions — replaying closed-loop demos open-loop **diverges** through
  contact-rich grasping (chaos amplifies tiny differences) and flings the cube.
  Validate from the **recorded observations** instead.
- **A success criterion satisfiable mid-action teaches an incomplete skill.** Our
  first criterion (`cube z < 0.05`) was true while the cube was still *gripped*,
  so episodes (and demos) ended before the release — the policy never learned to
  open the gripper. Fix: require the cube **settled** (`z < 0.03`, only true after
  release). The criterion *defines* what the policy learns.

---

## 5. File-by-file reference

### Scene / model
- **`mujoco_menagerie/robotstudio_so101/so101.xml`** — the SO-101 arm model (from
  menagerie; we don't edit it).
- **`mujoco_menagerie/robotstudio_so101/scene_table.xml`** — our scene: includes
  `so101.xml`, adds a table, a free-jointed cube, and a green `target_marker`
  site (repositioned each episode in goal-conditioned mode).

### Environment & expert
- **`so101_env.py`** — `SO101PickPlaceEnv(gym.Env)`. Loads the scene; defines
  `action_space` (6 joint targets from `ctrlrange`) and `observation_space`
  (18, or **20** goal-conditioned). `reset()` randomizes cube (polar sampling to
  match the arm's reachable annulus) and, if goal-conditioned, the target +
  marker. `step()` clips & applies the action, steps physics, tracks `_was_lifted`.
  `_is_success()` defines the task: **lifted + at target + settled (released)**.
  Modes: `task="pick_place"` (full) or `"pick_lift"` (grasp only);
  `goal_conditioned=True/False`.
- **`scripted_controller.py`** — `ScriptedPickPlaceController`: the IK + state-
  machine **expert**. `act(obs, info)` runs the FSM, sets an IK target, calls
  `mink.solve_ik`, returns 6 joint targets. Key constants: gripper open/close,
  `LIFT_Z` (ABSOLUTE lift height — must not be relative to the rising cube),
  `PLACE_HOVER_Z`, `PLACE_DOWN_Z`. Same interface as the learned policy.

### Data pipeline
- **`collect_demos.py`** — runs the expert across seeds; saves each SUCCESS as
  `demos/episode_NNNN.npz` (observations, actions, rewards, cube_start, seed).
  Discards failures. Prints the expert success rate.
- **`validate_demos.py`** — quality-checks demos from their RECORDED observations
  (peak cube height = clean low lift?, final placement precision, action
  smoothness). NOT open-loop replay (that diverges).
- **`inspect_demo.py`** — pretty-print one `.npz` (shapes, obs slices, rewards).
- **`replay_demo.py`** — visually replay a saved demo's actions in the viewer
  (note: open-loop, so it can diverge on contact-heavy demos; mainly illustrative).

### ACT (the final model)
- **`act_dataset.py`** — `ACTDataset`: loads all demos, **decimates to 50 Hz**
  (`decimation=4`), builds `(obs → next-K-actions)` chunk pairs, computes obs &
  action normalization stats. `save_stats()` → `act_stats.npz`.
- **`act_model.py`** — **defines the network** `ACTPolicy` (transformer encoder-
  decoder + CVAE) and `kl_divergence()`. The architecture only. (Engine.)
- **`act_train.py`** — the **training loop**: L1(chunk) + β·KL, AdamW, validation
  split, early stopping (`--patience`), saves best weights → `act_policy.pt`.
  Flags: `--epochs --batch --beta --decim --chunk --dmodel --patience --cpu`.
- **`act_policy.py`** — `ACTController`: **wraps the trained model for inference**.
  Loads `act_policy.pt` + `act_stats.npz`; normalizes obs; runs `z=0`; **temporal
  ensembling** + **decimation** (hold each action 4 steps); `act(obs)→action`.
  Same interface as the scripted expert. (Car around the engine.)
- **`eval_act.py`** — runs the policy on fresh seeds (1000+), strict metric;
  prints success rate. Flags: `--render`, `--pick-lift`, `--no-ensemble`,
  `--lookahead N`. In render mode, lingers after success so you see the cube land.

### Artifacts
- **`act_policy.pt`** — trained network weights.
- **`act_stats.npz`** — obs/action mean+std, chunk size, decimation. Needed at
  inference to normalize identically to training.
- **`demos/`** — the recorded demonstrations.

### Earlier stepping-stones (kept for reference, NOT in the final pipeline)
- **`bc_*.py`** — vanilla behavior cloning with a plain MLP (the 0% baseline).
- **`chunk_*.py`** — MLP + action chunking + temporal ensembling (pre-ACT).
- **`pick_and_place.py`** — the original standalone scripted task (before the env).
- **`archive/`** — Phase 1–2 MuJoCo learning scratch (pendulum, raw SO-101, IK demos).

---

## 6. The journey: what we tried and learned

Results are on the **strict** metric (real lift required; later, release required),
on fresh seeds the policy never trained on.

| Step | Result | What it taught |
|---|---|---|
| Scripted expert (IK + FSM) | ~97% | A hand-coded baseline that works; the demo generator. |
| Vanilla BC (MLP) | **0%** | The "47%" with a loose metric was lucky nudges. Real grasp = 0%. Felt distribution shift + multimodality. |
| MLP + chunking + ensembling | **0%** | Better behavior but still 0% on real pick-place. Simple BC has a ceiling here. |
| Hand-rolled ACT @ 200 Hz | **0%** | Diagnosed the **copycat trap**: 200 Hz absolute-target cloning ⇒ "predict current pose" ⇒ arm stalls. |
| ACT @ **50 Hz** (decimation) | **13%** | The rate fix unlocked real successes. First working learned pick-place. |
| ACT @ 50 Hz, **goal-conditioned** + clean demos | **83%** | Putting the target in the obs fixed the weak placing. |
| + **release-required** success, retrained | **86.7%** | Complete pick-place-AND-release; the cube is genuinely set down. |

Along the way we also fixed the **expert**: the place put the cube in the air
(used the held cube's height), a one-step-lag let states skip, RELEASE rose while
gripping (flung the cube), and — the big one — **LIFT used `cube_z + offset`, a
runaway target that fully extended the arm** (the "why is it straightening the
arm?" observation). Switching to an **absolute** lift height took the expert from
75% → 99% and placement from ~3.7 cm → ~0.75 cm.

---

## 7. Key lessons

1. **Clone at the right control rate (~30–50 Hz), not the sim's 200 Hz.** High-rate
   absolute-target cloning ⇒ copycat stall.
2. **Put the goal in the observation.** A policy can't reliably hit a target it
   can't see; memorizing a constant is fragile.
3. **Validate demos before training** — and from recorded observations, NOT
   open-loop replay (which diverges through contacts).
4. **The success criterion defines the learned skill.** If it's satisfiable mid-
   action, the policy learns a truncated behavior (never released the cube).
5. **Scripted experts: use absolute targets**, not targets relative to a moving
   grasped object (those run away).
6. **A loose metric hides a broken policy.** Always check the metric measures what
   you actually want; watch the rendered behavior.
7. **Low validation loss ≠ task success.** Closed-loop behavior is the real test
   (distribution shift, copycat, etc. are invisible to the loss).
8. **Don't fly blind on long jobs.** Run with `python -u`, log to a file, monitor.

---

## 8. How to run everything

All commands from the project root, using the `warp` interpreter
(`%USERPROFILE%\anaconda3\envs\warp\python.exe`, written `PY` below).

```powershell
# 1. Collect demos with the scripted expert (goal-conditioned)
PY -u collect_demos.py            # -> demos/*.npz, prints expert success rate

# 2. Validate the demos BEFORE training
PY validate_demos.py              # should report ~100% clean, low lift, precise

# 3. Train ACT (GPU). Early stopping ends it on plateau.
PY -u act_train.py --epochs 150 --batch 512 --beta 10 --decim 4 --patience 15
#    -> act_policy.pt + act_stats.npz

# 4. Evaluate (headless) on fresh seeds, strict metric
PY -u eval_act.py                 # -> success rate

# 5. Watch it (viewer); lingers after success so you see the cube land
PY eval_act.py --render

# Extras
PY run_expert.py                  # watch the scripted expert
PY eval_act.py --pick-lift        # score grasp-only (isolates the grasp skill)
PY inspect_demo.py demos/episode_0000.npz
```

Tip: for long background runs, redirect to a log and tail it:
`PY -u act_train.py ... > act_train.log 2>&1` then watch `act_train.log`.

---

## 9. Current status & next steps

**Status:** Phase 4 (imitation learning) exit criterion MET — goal-conditioned
ACT does complete pick-place-release at **~86.7%** with random targets.

**Levers to push past 86.7%** (sim):
- More demos (e.g. 500); bigger chunk (K=100, the ACT-paper default).
- Revive the CVAE: lower β to ~0.1–1 (KL currently collapses) or anneal it.
- Add the **wrist camera** → vision-based policy (the authentic VLA setup).

**Bigger directions:**
- **Phase 5 — buy the SO-101** and do sim-to-real (we're in the planned "deploy
  what I already have" position).
- **LeRobot** — port to the real framework (ACT / Diffusion Policy / SmolVLA);
  bridges to hardware. Refs found: `EE5108-DigitalTwins/lerobot_mujoco_sim`
  (MuJoCo + SO-101 + ACT), `StoneT2000/lerobot-sim2real` (PPO sim2real).

---

## 10. Glossary

- **BC** — Behavior Cloning: supervised imitation (obs → action).
- **IL / RL** — Imitation Learning (copy demos) / Reinforcement Learning (learn
  from reward). We use IL.
- **ACT** — Action Chunking Transformer (our policy architecture).
- **CVAE** — Conditional Variational Auto-Encoder (the multimodality-handling part
  of ACT).
- **Chunk (K)** — the number of future actions predicted from one observation.
- **Temporal ensembling** — averaging overlapping chunk predictions at run time.
- **Decimation** — downsampling the control rate (200 Hz → 50 Hz here).
- **Copycat trap** — high-rate cloning where "predict current pose" is near-optimal.
- **Goal-conditioning** — putting the target in the observation.
- **Posterior / KL collapse** — the CVAE latent becomes unused (KL → 0).
- **FK / IK** — Forward / Inverse Kinematics.
- **`qpos` / `qvel`** — generalized positions / velocities (MuJoCo state).
- **Distribution shift** — policy enters states absent from the training demos.
- **Strict metric** — success requires a real grasp+lift (+release), not a nudge.
