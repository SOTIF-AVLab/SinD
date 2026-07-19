from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from Simulation_test_toolchain.core.state_utils import get_agent_world_pose


def compute_ttc_simple(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    max_ttc: float = 10.0,
) -> Tuple[float, Optional[np.ndarray]]:
    rel_pos = target_pos - ego_pos
    rel_vel = target_vel - ego_vel
    rel_speed_sq = float(np.dot(rel_vel, rel_vel))
    if rel_speed_sq < 1e-8:
        return max_ttc, None

    ttc = -float(np.dot(rel_pos, rel_vel)) / rel_speed_sq
    if ttc <= 0.0 or ttc > max_ttc:
        return max_ttc, None

    ego_future = ego_pos + ego_vel * ttc
    target_future = target_pos + target_vel * ttc
    if np.linalg.norm(ego_future - target_future) > 5.0:
        return max_ttc, None
    return ttc, (ego_future + target_future) / 2.0


def compute_min_ttc(
    obs,
    ego_idx: int,
    neighbor_radius: float = 50.0,
    max_ttc: float = 10.0,
) -> Tuple[float, Optional[np.ndarray], Optional[int]]:
    ego = get_agent_world_pose(obs, ego_idx)
    best_ttc = max_ttc
    best_point = None
    best_idx = None

    for idx in range(len(obs.agent_name)):
        if idx == ego_idx:
            continue
        other = get_agent_world_pose(obs, idx)
        dist = np.linalg.norm(other["position"] - ego["position"])
        if dist > neighbor_radius:
            continue
        ttc, point = compute_ttc_simple(
            ego["position"],
            ego["velocity"],
            other["position"],
            other["velocity"],
            max_ttc=max_ttc,
        )
        if ttc < best_ttc:
            best_ttc = ttc
            best_point = point
            best_idx = idx

    return best_ttc, best_point, best_idx

