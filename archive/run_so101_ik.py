"""
Phase 2: Inverse Kinematics with mink.

Instead of manually setting joint angles, we give the gripper a target
XYZ position and mink figures out the joint angles automatically.

mink works by iteratively solving:
    J * dq = dx
where J is the Jacobian (how joint velocities map to end-effector velocity),
dq is the joint angle update, dx is the error toward the target.
It repeats this until the gripper is close enough to the target.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
import mink

SCENE = "mujoco_menagerie/robotstudio_so101/scene.xml"

model = mujoco.MjModel.from_xml_path(SCENE)
data  = mujoco.MjData(model)

# mink wraps the model into a Configuration object that tracks joint state
configuration = mink.Configuration(model)

# Define the IK task: drive the "gripperframe" site to a target pose
end_effector_task = mink.FrameTask(
    frame_name="gripperframe",
    frame_type="site",
    position_cost=1.0,      # how hard to match XYZ position
    orientation_cost=0.0,   # ignore orientation for now — just match position
)

# A few target positions to cycle through (world XYZ in meters)
# Tip: the arm base is at world origin (0,0,0), arm reaches ~0.3m out
targets = [
    np.array([0.15,  0.10, 0.15]),   # front-left, mid height
    np.array([0.20, -0.10, 0.20]),   # front-right, higher
    np.array([0.25,  0.00, 0.05]),   # straight ahead, low (near table height)
    np.array([0.10,  0.15, 0.25]),   # left, high
]

target_idx   = 0
target_start = 0.0
hold_sec     = 3.0

# IK solver tolerance and max iterations per sim step
solver   = "quadprog"
pos_threshold = 0.01  # metres — close enough

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_wall = time.time()

    while viewer.is_running():

        # --- IK: compute joint angles that move gripper toward current target ---
        target_pos = targets[target_idx]

        # Build a SE3 target (position only, identity orientation)
        target_pose = mink.SE3.from_translation(target_pos)
        end_effector_task.set_target(target_pose)

        # Sync mink's view of joint state from mjData
        configuration.update(data.qpos)

        # Solve IK → get velocity (dq) to apply this step
        vel = mink.solve_ik(
            configuration,
            [end_effector_task],
            model.opt.timestep,
            solver=solver,
            damping=1e-3,
        )

        # Integrate: new joint angles = old + dq * dt
        configuration.integrate_inplace(vel, model.opt.timestep)

        # Push the IK solution into data.ctrl (position actuators track this)
        data.ctrl[:6] = configuration.q[:6]

        mujoco.mj_step(model, data)
        viewer.sync()

        # --- Advance target after hold_sec or when close enough ---
        ee_pos = data.site_xpos[model.site("gripperframe").id]
        dist   = np.linalg.norm(ee_pos - target_pos)

        if data.time - target_start >= hold_sec or dist < pos_threshold:
            target_idx   = (target_idx + 1) % len(targets)
            target_start = data.time
            print(f"\nt={data.time:.1f}s  → new target: {targets[target_idx]}")

        if data.time % 0.5 < model.opt.timestep:
            print(f"t={data.time:.2f}  EE: {np.round(ee_pos,3)}  dist={dist:.3f}m")

        # Real-time pacing
        if data.time > time.time() - start_wall:
            time.sleep(data.time - (time.time() - start_wall))
