"""
Render-test for the vision policy: capture the external camera view and save
it, so we can verify the camera frames the workspace (arm + cube + green
target) before wiring it into the observation.

Saves both a large preview (for us to look at) and the actual policy-size crop.
"""

import numpy as np
import mujoco
from PIL import Image
from so101_env import SO101PickPlaceEnv

env = SO101PickPlaceEnv(render_mode=None, max_steps=2500,
                        randomize_cube=True, goal_conditioned=True)
obs, info = env.reset(seed=3)

# Step a little so the cube settles and the arm is at home
for _ in range(50):
    env.step(env.action_space.sample() * 0)   # zero action = hold

model, data = env.model, env.data

# Big preview to eyeball framing
big = mujoco.Renderer(model, height=480, width=640)
big.update_scene(data, camera="external")
Image.fromarray(big.render()).save("cam_external_preview.png")

# Policy-size image (what the network will actually see)
small = mujoco.Renderer(model, height=128, width=128)
small.update_scene(data, camera="external")
img = small.render()
Image.fromarray(img).save("cam_external_128.png")
# upscaled copy so we can eyeball the 128px content
Image.fromarray(img).resize((384, 384), Image.NEAREST).save("cam_external_128_big.png")

# Also the wrist camera, for reference
try:
    wr = mujoco.Renderer(model, height=480, width=640)
    wr.update_scene(data, camera="wrist_cam")
    Image.fromarray(wr.render()).save("cam_wrist_preview.png")
    print("wrist cam OK")
except Exception as e:
    print("wrist cam render failed:", e)

print(f"cube={info['cube_pos'][:2].round(3)} target={env.drop_target_xy.round(3)}")
print("saved: cam_external_preview.png (640x480), cam_external_96.png (96x96),"
      " cam_wrist_preview.png")
env.close()
