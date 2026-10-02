"""
Phase 1 starter: load a model, step the sim, render with the passive viewer.

This is the pattern you'll reuse everywhere. mjModel is the *static* description
of the world (parsed from XML, never changes). mjData is the *state* at the
current sim time (qpos = joint positions, qvel = velocities, ctrl = actuator
inputs, time = sim clock). mj_step advances mjData by one timestep using mjModel.
"""

import time
import mujoco
import mujoco.viewer

# 1. Load the model from XML. mjModel is read-only after this.
model = mujoco.MjModel.from_xml_path("archive/pendulum.xml")

# 2. Allocate state. mjData is what you'll read from and write to.
data = mujoco.MjData(model)

# 3. Start from the keyframe we defined in the XML so the pendulum actually moves.
mujoco.mj_resetDataKeyframe(model, data, 0)

# 4. Launch the passive viewer — it renders whatever's in `data` but does NOT
#    step the sim for you. We step it ourselves in the loop below.
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_wall = time.time()
    while viewer.is_running():
        # Step the physics. Default timestep is 0.002s (500 Hz) — check
        # model.opt.timestep if you want to change it.
        mujoco.mj_step(model, data)

        # Push the new state to the viewer window.
        viewer.sync()

        # Sleep so sim time tracks wall-clock time (otherwise it runs as fast
        # as your CPU allows and looks like a blur).
        sim_time = data.time
        wall_time = time.time() - start_wall
        if sim_time > wall_time:
            time.sleep(sim_time - wall_time)
        
        if data.time % 0.5 < model.opt.timestep:   # print every ~0.5 sim seconds
            print(f"t={data.time:.2f}  qpos={data.qpos}  qvel={data.qvel}")
