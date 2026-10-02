"""
Phase 3: Scripted pick-and-place on the SO-101.

State machine:
  HOME → OPEN_GRIP → HOVER → DESCEND → GRASP → LIFT → PLACE → RELEASE → HOME

IK drives the arm to each target position. Gripper is opened/closed by
commanding the last actuator (index 5).
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
import mink

SCENE  = "mujoco_menagerie/robotstudio_so101/scene_table.xml"

# ---------- helpers ----------

def get_cube_pos(data, model):
    return data.site_xpos[model.site("cube_center").id].copy()

def get_ee_pos(data, model):
    return data.site_xpos[model.site("gripperframe").id].copy()

def ee_error(data, model, target):
    return np.linalg.norm(get_ee_pos(data, model) - target)

# ---------- IK setup ----------

model = mujoco.MjModel.from_xml_path(SCENE)
data  = mujoco.MjData(model)

configuration = mink.Configuration(model)
ee_task = mink.FrameTask(
    frame_name="gripperframe",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=0.0,
)

# Posture task: keep joints near a target configuration. We'll use it
# to softly LOCK the wrist joints at a "point straight down" pose so IK
# plans around the constraint instead of fighting an external override.
posture_task = mink.PostureTask(model, cost=0.05)

# ---------- state machine ----------

GRIPPER_OPEN   =  1.5    # radians — jaw open
GRIPPER_CLOSED =  -0.1   # slight negative = clamp hard against the cube
HOVER_HEIGHT   =  0.08   # metres above cube centre
LIFT_HEIGHT    =  0.15   # metres above cube centre after grasp
DESCEND_OFFSET = -0.012  # metres RELATIVE to cube centre.
                         # NEGATIVE = gripperframe sinks BELOW cube centre,
                         # putting jaw tips near the table so the jaws fully
                         # wrap around the cube (cube half-size = 0.015).
                         # Tune in 2 mm steps if grasp keeps failing.

# Wrist orientation lock — pose that points gripper STRAIGHT DOWN at the table.
# We override these joints after IK so the wrist always approaches top-down,
# regardless of which arm config IK chose to reach the position target.
WRIST_FLEX_DOWN =  1.55   # POSITIVE tilts the SO-101 wrist DOWN (~pi/2) so the
                          # gripper faces the table. Verified empirically:
                          # at 1.55 the arm comes from above and grasps the cube.
WRIST_ROLL_FLAT =  0.0    # jaws aligned with world x-axis

# Table surface is at z=0; cube centre settles at z≈0.015
TABLE_Z  = 0.015
# Drop target: shift sideways on the same table surface
DROP_XY  = np.array([0.20, -0.13])

# Convergence tolerances — tighter = more accurate but slower transitions
TOL_LOOSE = 0.015   # for big moves (HOME, HOVER, LIFT, PLACE)
TOL_TIGHT = 0.008   # for grasping moves (DESCEND) — needs precision

states = [
    "HOME",
    "OPEN_GRIP",
    "HOVER",
    "DESCEND",
    "GRASP",
    "LIFT",
    "PLACE",
    "RELEASE",
]

state_idx   = 0
state_start = 0.0
ik_target   = np.array([0.15, 0.0, 0.15])   # initial safe position above table

# Home pose: arm upright, gripper open
HOME_CTRL = np.array([0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN])

data.ctrl[:] = HOME_CTRL

# Posture target is updated each iteration inside the loop so that
# non-wrist joints have zero posture error (and therefore no pull on them),
# while wrist joints get pulled toward WRIST_FLEX_DOWN / WRIST_ROLL_FLAT.

print("Starting pick-and-place. Initial state: HOME")

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_wall = time.time()

    while viewer.is_running():

        # ---- state logic ----
        current_state = states[state_idx]
        cube_pos      = get_cube_pos(data, model)
        ee_pos        = get_ee_pos(data, model)
        dist_to_ik    = ee_error(data, model, ik_target)
        time_in_state = data.time - state_start

        if current_state == "HOME":
            ik_target = np.array([0.15, 0.0, 0.30])
            if dist_to_ik < 0.015 or time_in_state > 2.0:
                state_idx += 1; state_start = data.time
                print(f"→ {states[state_idx]}")

        elif current_state == "OPEN_GRIP":
            data.ctrl[5] = GRIPPER_OPEN
            if time_in_state > 0.5:
                state_idx += 1; state_start = data.time
                print(f"→ {states[state_idx]}")

        elif current_state == "HOVER":
            ik_target = np.array([cube_pos[0], cube_pos[1],
                                  cube_pos[2] + HOVER_HEIGHT])
            if dist_to_ik < 0.012 or time_in_state > 3.0:
                state_idx += 1; state_start = data.time
                print(f"→ {states[state_idx]}")

        elif current_state == "DESCEND":
            # Drive gripperframe to cube centre (jaws straddle cube)
            ik_target = np.array([cube_pos[0], cube_pos[1],
                                  cube_pos[2] + DESCEND_OFFSET])
            if dist_to_ik < TOL_TIGHT or time_in_state > 4.0:
                state_idx += 1; state_start = data.time
                print(f"→ {states[state_idx]}")

        elif current_state == "GRASP":
            data.ctrl[5] = GRIPPER_CLOSED
            if time_in_state > 1.5:   # generous time for jaws to clamp
                state_idx += 1; state_start = data.time
                print(f"→ {states[state_idx]}")

        elif current_state == "LIFT":
            ik_target = np.array([cube_pos[0], cube_pos[1],
                                  cube_pos[2] + LIFT_HEIGHT])
            if dist_to_ik < 0.015 or time_in_state > 3.0:
                state_idx += 1; state_start = data.time
                print(f"→ {states[state_idx]}")

        elif current_state == "PLACE":
            ik_target = np.array([DROP_XY[0], DROP_XY[1],
                                  cube_pos[2] + HOVER_HEIGHT])
            if dist_to_ik < 0.015 or time_in_state > 4.0:
                state_idx += 1; state_start = data.time
                print(f"→ {states[state_idx]}")

        elif current_state == "RELEASE":
            # Open jaws AND retreat upward so gripper pulls away from the cube,
            # otherwise the cube lands on the wide fixed-jaw housing.
            data.ctrl[5] = GRIPPER_OPEN
            ik_target = np.array([DROP_XY[0], DROP_XY[1], 0.25])  # rise to 25 cm
            if time_in_state > 1.0:
                state_idx = 0; state_start = data.time   # loop back to HOME
                print(f"\n--- cycle complete, restarting ---\n→ {states[0]}")

        # ---- IK step ----
        configuration.update(data.qpos)
        ee_task.set_target(mink.SE3.from_translation(ik_target))

        # Posture target = current qpos with ONLY wrist joints overridden.
        # That way the posture cost has no effect on other joints (zero error)
        # but actively pulls the wrist toward the top-down pose.
        target_q = data.qpos.copy()
        target_q[3] = WRIST_FLEX_DOWN
        target_q[4] = WRIST_ROLL_FLAT
        posture_task.set_target(target_q)

        # Solve both tasks together. mink balances:
        #   FrameTask    → drive gripperframe to ik_target
        #   PostureTask  → keep wrist near WRIST_FLEX_DOWN, WRIST_ROLL_FLAT
        vel = mink.solve_ik(
            configuration,
            [ee_task, posture_task],
            model.opt.timestep,
            solver="quadprog",
            damping=1e-3,
        )
        configuration.integrate_inplace(vel, model.opt.timestep)

        # Hand the full IK solution to the actuators (no override needed)
        data.ctrl[:5] = configuration.q[:5]

        mujoco.mj_step(model, data)
        viewer.sync()

        # ---- logging ----
        if data.time % 1.0 < model.opt.timestep:
            print(f"  t={data.time:.1f}  state={current_state:<12}"
                  f"  EE={np.round(ee_pos,3)}  cube={np.round(cube_pos,3)}"
                  f"  dist={dist_to_ik:.3f}m")

        # real-time pacing
        elapsed = time.time() - start_wall
        if data.time > elapsed:
            time.sleep(data.time - elapsed)
