from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from Simulation_test_toolchain.core.state_utils import (
    get_agent_world_pose,
    scalar,
    to_numpy,
)
from .base import BasePolicy, PolicyAction, PolicyState
from .ttc_utils import compute_min_ttc


@dataclass
class RiskIDMState(PolicyState):
    velocity: float = 0.0
    acceleration: float = 0.0
    action_step: int = 0
    cached_acceleration: float = 0.0
    cached_ttc: float = 10.0
    cached_target_agent: Optional[str] = None
    has_cached_control: bool = False


def risk_idm_formula(
    ve: float,
    se: float,
    r: float,
    v_0: float = 15.0,
    a_max: float = 5.0,
    delta: float = 4.0,
    delta_t0: float = 2.0,
    t_headway: float = 0.5,
    a_comf: float = 2.0,
    a_min: float = -5.0,
) -> float:
    if not np.isfinite(ve):
        ve = 0.0
    if not np.isfinite(r):
        r = 10.0
    v_0 = max(v_0, 1e-6)
    r = max(r, 1e-6)
    r_star = delta_t0 + t_headway
    accel = a_max * (1.0 - np.power(ve / v_0, delta) - np.power(r_star / r, 2))
    if not np.isfinite(se):
        accel = a_max * (1.0 - np.power(ve / v_0, delta))
    return float(np.clip(accel, a_min, a_max))


class RiskIDMPolicy(BasePolicy):
    policy_name = "risk_idm"

    def __init__(
        self,
        dt: float = 0.1,
        desired_velocity: float = 15.0,
        max_acceleration: float = 5.0,
        min_acceleration: float = -5.0,
        neighbor_radius: float = 50.0,
        inference_interval_steps: int = 5,
        initial_velocity_override_mps: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(dt=dt)
        self.desired_velocity = desired_velocity
        self.max_acceleration = max_acceleration
        self.min_acceleration = min_acceleration
        self.neighbor_radius = neighbor_radius
        self.inference_interval_steps = max(1, int(inference_interval_steps))
        self.initial_velocity_override_mps = initial_velocity_override_mps

    def reset(self, obs, ego_idx: int = 0) -> RiskIDMState:
        info = get_agent_world_pose(obs, ego_idx)
        observed_velocity = float(np.linalg.norm(info["velocity"]))
        initial_velocity = observed_velocity
        if self.initial_velocity_override_mps is not None:
            override = float(self.initial_velocity_override_mps)
            if np.isfinite(override):
                initial_velocity = max(0.0, override)
        self.state = RiskIDMState(
            agent_name=obs.agent_name[ego_idx],
            dt=self.dt,
            initialized=True,
            velocity=initial_velocity,
        )
        self.last_command = {
            "policy": self.policy_name,
            "observed_initial_velocity": observed_velocity,
            "initial_velocity": initial_velocity,
            "initial_velocity_override_mps": self.initial_velocity_override_mps,
        }
        return self.state

    def get_action(self, obs, ego_idx: int = 0) -> PolicyAction:
        if self.state is None:
            self.reset(obs, ego_idx)

        should_infer = (
            not self.state.has_cached_control
            or self.state.action_step % self.inference_interval_steps == 0
        )
        if should_infer:
            info = get_agent_world_pose(obs, ego_idx)
            ttc, collision_point, target_idx = compute_min_ttc(
                obs, ego_idx, neighbor_radius=self.neighbor_radius
            )
            se = (
                float(np.linalg.norm(collision_point - info["position"]))
                if collision_point is not None
                else float("inf")
            )
            acceleration = risk_idm_formula(
                ve=self.state.velocity,
                se=se,
                r=ttc,
                v_0=self.desired_velocity,
                a_max=self.max_acceleration,
                a_min=self.min_acceleration,
            )
            target_agent = (
                str(obs.agent_name[target_idx]) if target_idx is not None else None
            )
            self.state.cached_acceleration = float(acceleration)
            self.state.cached_ttc = float(ttc)
            self.state.cached_target_agent = target_agent
            self.state.has_cached_control = True
        else:
            acceleration = float(self.state.cached_acceleration)
            ttc = float(self.state.cached_ttc)
            target_agent = self.state.cached_target_agent

        new_velocity = max(0.0, self.state.velocity + acceleration * self.dt)
        step_dist = new_velocity * self.dt
        next_pos, next_heading, path_points = _advance_along_gt_reference(
            obs, ego_idx, step_dist
        )

        self.state.velocity = new_velocity
        self.state.acceleration = acceleration
        self.state.action_step += 1
        self.last_command = {
            "policy": self.policy_name,
            "inference_interval_steps": self.inference_interval_steps,
            "used_cached_control": not should_infer,
            "acceleration": acceleration,
            "velocity": new_velocity,
            "initial_velocity_override_mps": self.initial_velocity_override_mps,
            "min_ttc": ttc,
            "target_agent": target_agent,
            "reference_progress_m": float(step_dist),
            "reference_path_points": int(path_points),
        }
        return PolicyAction(
            xyh=np.array([next_pos[0], next_pos[1], next_heading]),
            command=self.last_command.copy(),
        )


def _advance_along_gt_reference(obs, ego_idx: int, distance: float):
    pose = get_agent_world_pose(obs, ego_idx)
    curr_pos = np.asarray(pose["position"], dtype=float)
    curr_heading = float(pose["heading"])
    rot = np.asarray(pose["rotation"], dtype=float)

    points = [curr_pos]
    headings = [curr_heading]
    fut_len = 0
    if getattr(obs, "agent_fut_len", None) is not None:
        fut_len = int(to_numpy(obs.agent_fut_len[ego_idx]).reshape(-1)[0])
    if getattr(obs, "agent_fut", None) is not None:
        max_len = int(getattr(obs.agent_fut, "shape", [0, 0])[1])
        fut_len = min(fut_len, max_len)
        for fut_idx in range(fut_len):
            fut_state = obs.agent_fut[ego_idx, fut_idx]
            local_pos = to_numpy(fut_state.position)
            local_heading = scalar(fut_state.heading)
            world_pos = local_pos @ rot.T + curr_pos
            world_heading = curr_heading + local_heading
            if np.isfinite(world_pos).all() and np.isfinite(world_heading):
                points.append(np.asarray(world_pos[:2], dtype=float))
                headings.append(float(world_heading))

    if len(points) == 1 or distance <= 0.0:
        return curr_pos, curr_heading, len(points)

    points_arr = np.asarray(points, dtype=float)
    headings_arr = np.unwrap(np.asarray(headings, dtype=float))
    deltas = np.diff(points_arr, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    valid_seg = seg_lengths > 1e-6
    if not np.any(valid_seg):
        return curr_pos, curr_heading, len(points)

    cumulative = 0.0
    last_valid_idx = 0
    for seg_idx, seg_len in enumerate(seg_lengths):
        if seg_len <= 1e-6:
            continue
        last_valid_idx = seg_idx + 1
        if cumulative + seg_len >= distance:
            ratio = (distance - cumulative) / seg_len
            ratio = float(np.clip(ratio, 0.0, 1.0))
            next_pos = points_arr[seg_idx] + ratio * deltas[seg_idx]
            next_heading = headings_arr[seg_idx] + ratio * (
                headings_arr[seg_idx + 1] - headings_arr[seg_idx]
            )
            return next_pos, _wrap_angle(next_heading), len(points)
        cumulative += seg_len

    return points_arr[last_valid_idx], _wrap_angle(headings_arr[last_valid_idx]), len(points)


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
