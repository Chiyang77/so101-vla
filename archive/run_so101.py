"""
Phase 2: Load the real SO-101 from mujoco_menagerie and drive it
to a sequence of joint-space poses using position actuators.

Key difference from the pendulum:
  - Actuators are <position> type: data.ctrl[i] = target_angle (radians)
  - The built-in servo PD drives to that angle automatically each step
  - No manual PD controller needed — the model already has one tuned to
    the real STS3215 servo gains
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

SCENE = "mujoco_menagerie/robotstudio_so101/scene.xml"

model = mujoco.MjModel.from_xml_path(SCENE)
data  = mujoco.MjData(model)

# Print joint and actuator names so we know the order of data.ctrl
print("Joints:")
for i in range(model.njnt):
    print(f"  [{i}] {model.joint(i).name}")

print("\nActuators (ctrl index → name):")
for i in range(model.nu):
    print(f"  [{i}] {model.actuator(i).name}")

# --- Define a few target poses (all values in radians) ---
# Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

HOME = np.array([0.0,  0.0,  0.0,  0.0,  0.0,  0.0])   # straight up / rest
POSE_A = np.array([0.5, -0.8,  1.0, -0.5,  0.0,  0.5])  # reach forward-left
POSE_B = np.array([-0.5, -0.6,  0.8,  0.3,  1.0,  0.5]) # reach forward-right

poses    = [HOME, POSE_A, POSE_B, HOME]
hold_sec = 2.5   # seconds to hold each pose before moving to next

pose_idx    = 0
pose_start  = 0.0  # sim time when we started the current pose

# Apply first pose immediately
data.ctrl[:] = poses[0]

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_wall = time.time()

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

        # Advance to next pose after hold_sec sim-seconds
        if data.time - pose_start >= hold_sec:
            pose_idx  = (pose_idx + 1) % len(poses)
            pose_start = data.time
            data.ctrl[:] = poses[pose_idx]
            print(f"\nt={data.time:.1f}s → moving to pose {pose_idx}: {poses[pose_idx]}")

        # Print end-effector position every ~0.5 sim-seconds
        if data.time % 0.5 < model.opt.timestep:
            ee = data.site_xpos[model.site("gripperframe").id]
            print(f"t={data.time:.2f}  EE pos: {np.round(ee, 3)}")

        # Real-time pacing
        sim_time  = data.time
        wall_time = time.time() - start_wall
        if sim_time > wall_time:
            time.sleep(sim_time - wall_time)
