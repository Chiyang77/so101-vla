import time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('archive/pendulum_actuator.xml')

data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

# Target joint angles we want to hold (radians)
target_qpos = [0.5, -0.3]   # link1 leaning forward, link2 slightly bent

# PD gains  —  P pushes toward target, D damps oscillation
Kp = 0.8   # proportional gain
Kd = 0.1   # derivative gain

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_wall = time.time()
    while viewer.is_running():

        # PD controller: error = target - current, applied every step
        for i in range(2):
            error    = target_qpos[i] - data.qpos[i]
            d_error  = -data.qvel[i]            # we want velocity → 0
            data.ctrl[i] = Kp * error + Kd * d_error

        mujoco.mj_step(model, data)

        viewer.sync()

        sim_time = data.time
        wall_time = time.time() - start_wall
        if sim_time > wall_time:
            time.sleep(sim_time-wall_time)
        
        if data.time % 0.5 < model.opt.timestep:   # print every ~0.5 sim seconds
            print(f"t={data.time:.2f}  qpos={data.qpos}  qvel={data.qvel}")
            tip_pos = data.site_xpos[model.site("tip").id]
            print(f"tip: {tip_pos}")