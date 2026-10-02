"""
Render a vision-policy rollout as a montage of frames (so we can SEE the
behaviour). Runs the trained vision ACT on a given seed, captures the external
camera at higher resolution at intervals, and tiles them into one PNG.

Usage: python render_vision_demo.py <seed> <out.png>
"""
import sys
import numpy as np
import mujoco
from PIL import Image
from so101_env import SO101PickPlaceEnv
from act_policy_vision import VisionACTController

DECIM = 4
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1013
out  = sys.argv[2] if len(sys.argv) > 2 else "vision_demo.png"
NCOLS, NROWS = 4, 2
NFRAMES = NCOLS * NROWS
RES = 240

env = SO101PickPlaceEnv(obs_mode="state", img_size=128, max_steps=2500,
                        randomize_cube=True, goal_conditioned=True, task="pick_place")
policy = VisionACTController(ensemble=True)
big = mujoco.Renderer(env.model, height=RES, width=RES)

def proprio(): return np.concatenate([env.data.qpos[:6], env.data.qvel[:6]]).astype(np.float32)

_, info = env.reset(seed=seed); policy.reset()
frames, steps = [], []
t, done = 0, False
while t < env.max_steps and not done:
    vobs = {"image_ext": env._render_external(),
            "image_wrist": env._render_wrist(), "proprio": proprio()}
    action = policy.act(vobs, info)
    for _ in range(DECIM):
        _, r, term, trunc, info = env.step(action); t += 1
        if term or trunc: done = True; break
    big.update_scene(env.data, camera="external")
    frames.append(big.render()); steps.append(t)

# pick NFRAMES evenly spaced
idx = np.linspace(0, len(frames) - 1, NFRAMES).astype(int)
tiles = [Image.fromarray(frames[i]) for i in idx]
canvas = Image.new("RGB", (NCOLS * RES, NROWS * RES), (0, 0, 0))
for k, im in enumerate(tiles):
    canvas.paste(im, ((k % NCOLS) * RES, (k // NCOLS) * RES))
canvas.save(out)
print(f"seed={seed} success={info['success']} steps={t} frames_at={[steps[i] for i in idx]}")
print(f"saved {out}")
env.close()
