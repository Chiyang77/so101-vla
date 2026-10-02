"""
Phase 4 expert: scripted pick-and-place controller wrapped to fit the env's
controller interface.

It reuses the state machine + IK from pick_and_place.py, but instead of
running its own loop, it exposes:

    controller.reset()              # call at start of each episode
    action = controller.act(obs, info)   # call once per env.step

This is the same interface a learned policy will have later, so we can
swap one for the other without touching env code or the training loop.
"""

import numpy as np
import mink


class ScriptedPickPlaceController:
    """State-machine + IK expert policy for the SO-101 pick-and-place env."""

    STATES = [
        "HOME",
        "OPEN_GRIP",
        "HOVER",
        "DESCEND",
        "GRASP",
        "LIFT",
        "PLACE_HOVER",   # move ABOVE the target, cube still held high
        "PLACE_DOWN",    # descend so the cube is just above the table
        "RELEASE",       # open jaws (cube already low) + retreat up
    ]

    # Tunable constants — copied from pick_and_place.py
    GRIPPER_OPEN    =  1.5
    GRIPPER_CLOSED  = -0.3   # firmer clamp so the cube doesn't slip mid-transport
    HOVER_HEIGHT    =  0.08
    LIFT_Z          =  0.09   # ABSOLUTE gripper height after grasp (just clear the
                             # table). Must NOT be cube_z + offset — that runs away
                             # upward as the held cube rises, fully extending the arm.
    DESCEND_OFFSET  = -0.012
    PLACE_HOVER_Z   =  0.09    # low hover directly above the target (same height)
    PLACE_DOWN_Z    =  0.045   # lower the cube close to the table, then release
    WRIST_FLEX_DOWN =  1.55
    WRIST_ROLL_FLAT =  0.0

    def __init__(self, env):
        # Privileged access — needed for IK. A learned policy wouldn't need this.
        self.env   = env
        self.model = env.model
        self.data  = env.data

        # IK setup (same as pick_and_place.py)
        self.configuration = mink.Configuration(self.model)
        self.ee_task = mink.FrameTask(
            frame_name="gripperframe",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        self.posture_task = mink.PostureTask(self.model, cost=0.05)

        # Read task params from env so we stay in sync
        self.drop_target_xy = env.drop_target_xy

        self.reset()

    # ---------------------------------------------------------------------
    # Controller API
    # ---------------------------------------------------------------------

    def reset(self):
        """Reset internal state at the start of an episode."""
        self.state_idx   = 0
        self.state_start = 0.0
        self.ik_target   = np.array([0.15, 0.0, 0.30])
        # Re-read the drop target — it may be randomized per episode (goal-cond).
        self.drop_target_xy = self.env.drop_target_xy

        # last_action holds our current ctrl. Gripper open at episode start.
        self.last_action = np.zeros(6, dtype=np.float32)
        self.last_action[5] = self.GRIPPER_OPEN

    def act(self, obs, info):
        """Return the next 6-element action."""
        # Read privileged cube/ee positions from `info` (not the obs), so the
        # expert works regardless of obs_mode (state OR pixels). The expert is
        # allowed to be privileged — it's the demo generator, not the policy.
        cube_pos = info["cube_pos"]
        ee_pos   = info["ee_pos"]
        sim_time = self.data.time   # privileged read

        current_state = self.STATES[self.state_idx]
        time_in_state = sim_time - self.state_start

        # Distance to whatever target the CURRENT state is about to set.
        # We measure against the freshly-set ik_target (see _arrived) so a
        # state never falsely "arrives" using the previous state's target.
        def arrived(tol):
            d = np.linalg.norm(ee_pos - self.ik_target)
            return d < tol

        # ----- state machine -----
        if current_state == "HOME":
            self.ik_target = np.array([0.15, 0.0, 0.30])
            if arrived(0.015) or time_in_state > 2.0:
                self._advance(sim_time)

        elif current_state == "OPEN_GRIP":
            self.last_action[5] = self.GRIPPER_OPEN
            if time_in_state > 0.5:
                self._advance(sim_time)

        elif current_state == "HOVER":
            self.ik_target = np.array([cube_pos[0], cube_pos[1],
                                       cube_pos[2] + self.HOVER_HEIGHT])
            if arrived(0.012) or time_in_state > 3.0:
                self._advance(sim_time)

        elif current_state == "DESCEND":
            self.ik_target = np.array([cube_pos[0], cube_pos[1],
                                       cube_pos[2] + self.DESCEND_OFFSET])
            if arrived(0.008) or time_in_state > 4.0:
                self._advance(sim_time)

        elif current_state == "GRASP":
            self.last_action[5] = self.GRIPPER_CLOSED
            if time_in_state > 1.5:
                self._advance(sim_time)

        elif current_state == "LIFT":
            # Lift straight up to an ABSOLUTE height (not relative to the rising
            # cube — that was the runaway that extended the arm).
            self.ik_target = np.array([cube_pos[0], cube_pos[1], self.LIFT_Z])
            if arrived(0.015) or time_in_state > 3.0:
                self._advance(sim_time)

        elif current_state == "PLACE_HOVER":
            # Move ABOVE the target at a fixed safe height (absolute z, NOT
            # relative to the held cube — that was the old bug that left the
            # cube up in the air).
            self.ik_target = np.array([self.drop_target_xy[0],
                                       self.drop_target_xy[1],
                                       self.PLACE_HOVER_Z])
            if arrived(0.015) or time_in_state > 4.0:
                self._advance(sim_time)

        elif current_state == "PLACE_DOWN":
            # Lower the held cube to low over the table at the target.
            self.ik_target = np.array([self.drop_target_xy[0],
                                       self.drop_target_xy[1],
                                       self.PLACE_DOWN_Z])
            if arrived(0.015) or time_in_state > 4.0:
                self._advance(sim_time)

        elif current_state == "RELEASE":
            # Cube is low at the target. Open the jaws but HOLD the gripper in
            # place (don't rise yet) so the cube drops straight down and settles
            # instead of being carried/flung as the arm moves away.
            self.last_action[5] = self.GRIPPER_OPEN
            self.ik_target = np.array([self.drop_target_xy[0],
                                       self.drop_target_xy[1],
                                       self.PLACE_DOWN_Z])
            if time_in_state > 0.8:
                # Loop back to HOME (which retreats up/back) — env will have
                # terminated already if the placement succeeded.
                self.state_idx   = 0
                self.state_start = sim_time

        # ----- IK step -----
        self.configuration.update(self.data.qpos)
        self.ee_task.set_target(mink.SE3.from_translation(self.ik_target))

        # Posture target = current qpos with only wrist joints overridden
        target_q = self.data.qpos.copy()
        target_q[3] = self.WRIST_FLEX_DOWN
        target_q[4] = self.WRIST_ROLL_FLAT
        self.posture_task.set_target(target_q)

        vel = mink.solve_ik(
            self.configuration,
            [self.ee_task, self.posture_task],
            self.model.opt.timestep,
            solver="quadprog",
            damping=1e-3,
        )
        self.configuration.integrate_inplace(vel, self.model.opt.timestep)

        # Arm joints from IK, gripper from state machine logic
        self.last_action[:5] = self.configuration.q[:5]
        return self.last_action.copy()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _advance(self, sim_time):
        self.state_idx  += 1
        self.state_start = sim_time
