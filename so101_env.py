"""
Phase 3 finale: SO-101 pick-and-place wrapped as a Gymnasium environment.

This is the bridge between the scripted state machine (Phase 3) and
learned policies (Phase 4). Any controller — scripted, RL, imitation
learning — can drive this env through the standard Gymnasium interface:

    obs, info = env.reset()
    while not done:
        action = my_controller(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer


class SO101PickPlaceEnv(gym.Env):
    """Pick up a cube on a table and place it at a target XY."""

    metadata = {"render_modes": ["human"], "render_fps": 200}

    SCENE = "mujoco_menagerie/robotstudio_so101/scene_table.xml"

    def __init__(self, render_mode=None, max_steps=2000, randomize_cube=True,
                 task="pick_place", goal_conditioned=False,
                 obs_mode="state", img_size=128):
        # task: "pick_place" = grasp, lift, AND place at the drop target (strict).
        #       "pick_lift"  = success once the cube is grasped and held up,
        #                       isolating the contact-rich grasp skill.
        # goal_conditioned: if True, the drop target is RANDOMIZED each episode
        #       and appended to the observation (obs 18 -> 20) so the policy is
        #       TOLD where to place instead of memorizing a fixed location.
        # obs_mode: "state"  -> flat privileged vector (qpos,qvel,cube,ee[,target]).
        #           "pixels" -> {"image": HxWx3 uint8 from the external camera,
        #                        "proprio": [qpos[:6], qvel[:6]] (12,)}. The cube
        #                        and target must be perceived from the image (the
        #                        green marker is visible), so nothing privileged.
        self.goal_conditioned = goal_conditioned
        self.obs_mode = obs_mode
        self.img_size = img_size
        self._renderer = None   # created lazily on first pixel render
        # --- Load the world ---
        self.model = mujoco.MjModel.from_xml_path(self.SCENE)
        self.data = mujoco.MjData(self.model)

        self.task = task
        self.lift_success_height = 0.10   # cube centre this high = grasped & lifted
        self.max_steps = max_steps
        self.randomize_cube = randomize_cube
        self.render_mode = render_mode
        self.viewer = None
        self.step_count = 0

        # --- Action space: 6 joint targets bounded by each actuator's ctrlrange ---
        # actuator_ctrlrange is (nu, 2) — low/high per actuator
        ctrl_low  = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(low=ctrl_low, high=ctrl_high, dtype=np.float32)

        # --- Observation space ---
        if obs_mode == "pixels":
            # {image: HxWx3 uint8, proprio: [qpos[:6], qvel[:6]] float32}
            self.observation_space = spaces.Dict({
                "image":   spaces.Box(0, 255, (img_size, img_size, 3), np.uint8),
                "proprio": spaces.Box(-np.inf, np.inf, (12,), np.float32),
            })
        else:
            # base: [qpos[:6], qvel[:6], cube_pos[3], ee_pos[3]] = 18
            # goal-conditioned: + target_xy[2] = 20
            obs_dim = 20 if goal_conditioned else 18
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

        # --- Cache site & joint IDs (one-time name->index lookup) ---
        self.cube_site_id   = self.model.site("cube_center").id
        self.ee_site_id     = self.model.site("gripperframe").id
        self.target_site_id = self.model.site("target_marker").id
        # Address into qpos where the cube's freejoint lives (7 entries: xyz + quat)
        cube_joint_id = self.model.joint("cube_joint").id
        self.cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]

        # Drop target — where we want the cube to end up
        self.drop_target_xy = np.array([0.20, -0.13], dtype=np.float32)
        self.success_radius = 0.04   # 4 cm tolerance for "successfully placed"
        # Cube must rise above this height at some point to prove a REAL grasp
        # (cube rests at z~0.015; this requires it to be genuinely lifted).
        self.lift_height = 0.06

    # ---------------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Start a new episode. Returns (initial_obs, info)."""
        super().reset(seed=seed)

        # Wipe physics state (joints, velocities, contacts, time) to defaults
        mujoco.mj_resetData(self.model, self.data)

        # Randomise the cube's starting XY on the table.
        # Use POLAR sampling to match the arm's annular workspace:
        #   - too close to base → IK folds → grasp unreliable
        #   - too far → arm singular → IK unstable
        # The "good zone" is roughly r in [0.20, 0.27] m, theta in [-0.4, 0.4] rad
        if self.randomize_cube:
            MIN_R, MAX_R = 0.22, 0.27         # radial distance from arm base
            MAX_THETA    = 0.4                 # half-angle in radians (~±23°)
            r     = self.np_random.uniform(MIN_R, MAX_R)
            theta = self.np_random.uniform(-MAX_THETA, MAX_THETA)
            cube_x = r * np.cos(theta)
            cube_y = r * np.sin(theta)
        else:
            cube_x, cube_y = 0.25, 0.0

        # Write the cube's freejoint qpos directly: [x, y, z, qw, qx, qy, qz]
        addr = self.cube_qpos_addr
        self.data.qpos[addr:addr+3] = [cube_x, cube_y, 0.02]   # 2 cm above table → settles
        self.data.qpos[addr+3:addr+7] = [1.0, 0.0, 0.0, 0.0]    # identity quaternion

        # Drop target: randomized per episode in goal-conditioned mode.
        if self.goal_conditioned:
            r     = self.np_random.uniform(0.22, 0.27)
            theta = self.np_random.uniform(-0.4, 0.4)
            self.drop_target_xy = np.array([r*np.cos(theta), r*np.sin(theta)],
                                           dtype=np.float32)
        # Move the visual marker disk to the target (visualization only).
        self.model.site_pos[self.target_site_id][:2] = self.drop_target_xy

        # Open jaws, arm at zero pose
        self.data.ctrl[:] = 0.0
        self.data.ctrl[5] = 1.5   # gripper open

        # Forward kinematics to refresh derived state (site_xpos, etc.)
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self._was_lifted = False   # has the cube been genuinely lifted this episode?
        return self._get_obs(), self._get_info()

    def step(self, action):
        """Apply action, advance one timestep, return SARS tuple."""
        # Clip action to the legal range (in case the controller produces out-of-range values)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        mujoco.mj_step(self.model, self.data)

        # Track whether the cube was ever genuinely lifted off the table.
        # This is what distinguishes a real pick-and-place from a lucky nudge.
        if self._cube_pos()[2] > self.lift_height:
            self._was_lifted = True

        if self.viewer is not None:
            self.viewer.sync()

        obs        = self._get_obs()
        reward     = self._get_reward()
        terminated = self._is_success()
        truncated  = self.step_count >= self.max_steps
        info       = self._get_info()

        self.step_count += 1
        return obs, reward, terminated, truncated, info

    def render(self):
        """Open the MuJoCo viewer (lazy — only first call)."""
        if self.render_mode == "human" and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _cube_pos(self):
        return self.data.site_xpos[self.cube_site_id].copy()

    def _ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def _render_cam(self, name):
        """Render a named camera → (img_size, img_size, 3) uint8."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.img_size,
                                             width=self.img_size)
        self._renderer.update_scene(self.data, camera=name)
        return self._renderer.render()

    def _render_external(self):
        return self._render_cam("external")

    def _render_wrist(self):
        return self._render_cam("wrist_cam")

    def _get_obs(self):
        if self.obs_mode == "pixels":
            proprio = np.concatenate([self.data.qpos[:6],
                                      self.data.qvel[:6]]).astype(np.float32)
            return {"image": self._render_external(), "proprio": proprio}
        parts = [
            self.data.qpos[:6],   # arm + gripper joint angles
            self.data.qvel[:6],   # joint velocities
            self._cube_pos(),     # cube XYZ in world
            self._ee_pos(),       # gripper XYZ in world
        ]
        if self.goal_conditioned:
            parts.append(self.drop_target_xy)   # where to place (2 numbers)
        return np.concatenate(parts).astype(np.float32)

    def _get_reward(self):
        """Dense shaped reward: pull EE to cube, push cube to target.

        Reward components:
          -0.1 * distance(EE, cube)        — encourage approaching the cube
          -0.1 * distance(cube_xy, target) — encourage moving cube toward goal
          +50  on success                  — big bonus for placing it correctly
        """
        cube = self._cube_pos()
        ee   = self._ee_pos()

        d_ee_to_cube = np.linalg.norm(ee - cube)
        d_cube_to_target = np.linalg.norm(cube[:2] - self.drop_target_xy)

        reward = -0.1 * d_ee_to_cube - 0.1 * d_cube_to_target
        if self._is_success():
            reward += 50.0
        return float(reward)

    def _is_success(self):
        """Real pick-and-place: cube was LIFTED, then RELEASED and settled at
        the target on the table.

        - self._was_lifted rejects "lucky nudge" episodes (never grasped/raised).
        - The settled-height check (cube_z < 0.03) requires the cube to be
          actually RESTING on the table, i.e. released by the gripper. A gripped
          cube sits at the place-down height (~0.045+), so this only fires after
          the jaws open and the cube drops — the full place-AND-release.
        """
        cube = self._cube_pos()
        if self.task == "pick_lift":
            # Success as soon as the cube is grasped and held up.
            return bool(cube[2] > self.lift_success_height)
        at_target = np.linalg.norm(cube[:2] - self.drop_target_xy) < self.success_radius
        settled   = cube[2] < 0.03   # resting on table => released, not gripped
        return bool(at_target and settled and self._was_lifted)

    def _get_info(self):
        """Auxiliary info — not used for learning but useful for logging/debug."""
        return {
            "cube_pos": self._cube_pos(),
            "ee_pos":   self._ee_pos(),
            "success":  self._is_success(),
            "step":     self.step_count,
        }
