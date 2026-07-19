from __future__ import annotations

from typing import Any, Dict

import numpy as np


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _agent_sequence(sequence: Any, agent_idx: int) -> Any:
    return sequence[agent_idx]


def _xy_at(sequence: Any, timestep: int) -> np.ndarray:
    value = sequence[timestep]
    if hasattr(value, "position"):
        return to_numpy(value.position)[:2]
    return to_numpy(value)[:2]


def get_agent_world_pose(obs, agent_idx: int) -> Dict[str, np.ndarray | float]:
    if hasattr(obs, "agents_from_world_tf") and obs.agents_from_world_tf is not None:
        agent_from_world = to_numpy(obs.agents_from_world_tf[agent_idx])
        world_from_agent = np.linalg.inv(agent_from_world)
        pos = world_from_agent[:2, 2]
        heading = float(np.arctan2(world_from_agent[1, 0], world_from_agent[0, 0]))
        rot = world_from_agent[:2, :2]
    else:
        state = obs.curr_agent_state[agent_idx]
        pos = to_numpy(state.position)
        heading = scalar(state.heading)
        rot = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )

    velocity = np.zeros(2)
    if getattr(obs, "agent_hist", None) is not None and obs.agent_hist_len[agent_idx] > 1:
        hist_len = int(obs.agent_hist_len[agent_idx].item())
        agent_hist = _agent_sequence(obs.agent_hist, agent_idx)
        local_recent = _xy_at(agent_hist, hist_len - 1)
        local_prev = _xy_at(agent_hist, hist_len - 2)
        dt = float(obs.dt[agent_idx].item()) if hasattr(obs, "dt") else 0.1
        velocity = rot @ ((local_recent - local_prev) / dt)
        if not np.isfinite(velocity).all():
            velocity = np.zeros(2)

    extent = np.array([4.2, 1.8, 1.5])
    if getattr(obs, "agent_hist_extent", None) is not None:
        extent = to_numpy(obs.agent_hist_extent[agent_idx, -1])

    return {
        "position": pos,
        "heading": heading,
        "rotation": rot,
        "velocity": velocity,
        "extent": np.nan_to_num(extent, nan=1.5, posinf=4.2, neginf=1.5),
    }


def get_ground_truth_next_xyh(obs, agent_idx: int) -> np.ndarray:
    pose = get_agent_world_pose(obs, agent_idx)
    curr_pos = pose["position"]
    curr_heading = float(pose["heading"])
    rot = pose["rotation"]

    if obs.agent_fut_len[agent_idx] >= 1:
        fut_pos_local = to_numpy(obs.agent_fut[agent_idx, 0].position)
        fut_heading = scalar(obs.agent_fut[agent_idx, 0].heading)
        next_pos = fut_pos_local @ rot.T + curr_pos
        next_heading = curr_heading + fut_heading
    else:
        next_pos = curr_pos
        next_heading = curr_heading

    return np.array([next_pos[0], next_pos[1], next_heading], dtype=float)


def xyh_to_state_array(xyh: np.ndarray):
    from trajdata.data_structures.state import StateArray

    state = np.array([xyh[0], xyh[1], 0.0, xyh[2]], dtype=float)
    return StateArray.from_array(state, "x,y,z,h")
