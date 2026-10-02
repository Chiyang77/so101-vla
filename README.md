# VLA — SO-101 pick-and-place (sim-first)

Learning project: train manipulation policies in MuJoCo, then deploy to a real
SO-ARM101. See `plan.md` for the phased roadmap and current status.

## Documentation
- **[docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md)** — deep living notes: every
  file's purpose, the knowledge behind each piece, the full journey, lessons,
  and how to run everything. **Start here.**
- **[docs/CVAE_explained.md](docs/CVAE_explained.md)** — deep dive on the CVAE.
- `plan.md` — phased roadmap and status log.

## Setting up from a fresh clone

Two third-party repos are not stored here (see `.gitignore`); clone them into the project root:

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
copy assets\scene_table.xml mujoco_menagerie\robotstudio_so101\scene_table.xml
```

`assets/scene_table.xml` is our scene (SO-101 + table + cube + target + cameras).

The real-arm code runs on Hiwonder's LeRobot snapshot (huggingface/lerobot commit
`882c80d4` plus Hiwonder's patches), installed editable at `lerobot/lerobot`.
`patches/lerobot_local_changes.patch` holds every local change to it: Hiwonder's
patches (motors_bus.py, cameras/utils.py, configs/policies.py) plus ours
(Windows checkpoint-link fallback in utils/train_utils.py, `GRIPPER_SNAP` in record.py).
Re-apply with `git -C lerobot/lerobot apply ../../patches/lerobot_local_changes.patch`.

Not stored: model weights (`*.pt`, `outputs/`), datasets (`demos*/`, and LeRobot
datasets in `~/.cache/huggingface/lerobot`). Robot calibration backups are in
`calibration_backup/`.

## Environment

Sim/ML scripts use the `warp` conda env (torch + mujoco + mink + gymnasium);
real-arm work uses the `lerobot` conda env:

```
%USERPROFILE%\anaconda3\envs\warp\python.exe <script>.py
```

Always run from the project root (paths like `mujoco_menagerie/...` are root-relative).

## Real-arm tools (Phase 5)
| File | Role |
|------|------|
| `go_home.py` | Drive the follower to the demo start pose (run before each eval episode) |
| `run_eval.bat` | Eval loop: place ball -> go home -> record one policy episode |
| `relabel_gripper.py` | Copy a dataset with the gripper action relabeled from the measured jaw |
| `docs/lerobot_commands.txt` | Teleop / calibrate / record / train commands used |

## File map

### Core (Phase 3) — the environment & scripted expert
| File | Role |
|------|------|
| `mujoco_menagerie/robotstudio_so101/scene_table.xml` | SO-101 + table + cube scene |
| `so101_env.py` | Gymnasium env (reset/step, obs, reward, success check) |
| `scripted_controller.py` | Hand-coded expert: state machine + IK (the demo generator) |
| `pick_and_place.py` | Original standalone scripted task (superseded by the two above) |
| `test_env.py` | Smoke-test the env with random actions |

### Data pipeline (Phase 4) — generate & inspect demos
| File | Role |
|------|------|
| `collect_demos.py` | Run the expert across seeds, save successes to `demos/*.npz` |
| `run_expert.py` | Watch the scripted expert in the viewer |
| `eval_expert.py` | Measure the expert's success rate (headless) |
| `replay_seed.py` | Re-run one seed with viewer + per-state logging (debug failures) |
| `inspect_demo.py` | Print the contents of a saved `demos/*.npz` |
| `demos/` | 200 recorded successful trajectories (the training data) |

### Behavior cloning — vanilla MLP (Phase 4, baseline)
| File | Role |
|------|------|
| `bc_dataset.py` | Pool all (obs, action) pairs; compute normalization stats |
| `bc_model.py` | MLP: 18 obs -> 6 action |
| `bc_train.py` | Supervised training loop (`--cpu` to force CPU) |
| `bc_policy.py` | Wrap trained net as a controller (act interface) |
| `eval_bc.py` | Measure BC policy success rate |
| `bc_policy.pt`, `bc_stats.npz` | Trained weights + normalization stats |

### Action chunking + temporal ensembling (Phase 4, ACT-style)
| File | Role |
|------|------|
| `chunk_dataset.py` | Targets are chunks of K=50 future actions |
| `chunk_model.py` | MLP: 18 obs -> 50x6 actions |
| `chunk_train.py` | Training loop for the chunking model |
| `chunk_policy.py` | Receding-horizon OR temporal-ensemble inference |
| `eval_chunk.py` | Measure chunk policy (`--ensemble`, `--horizon N`, `--render`) |
| `chunk_policy.pt`, `chunk_stats.npz` | Trained weights + stats |

### archive/ — Phase 1–2 learning scratch (standalone, run from project root)
Pendulum + raw SO-101 / IK scripts used while learning MuJoCo fundamentals.
Run e.g. `python archive/run_pendulum.py` from the project root.

## Current status (see plan.md)

Honest result with a strict success metric (cube must be genuinely lifted):
- Scripted expert: ~97%
- Vanilla BC and chunking+ensembling MLPs: ~0% on TRUE pick-and-place
  (earlier "47%" was lucky nudges; the loose metric masked it)

Conclusion: vanilla BC can't do the contact-rich grasp. Next step is LeRobot
(real ACT transformer+CVAE / Diffusion Policy), which also bridges to the real arm.
