from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from Simulation_test_toolchain.core.config import ToolchainConfig
from Simulation_test_toolchain.core.records import AgentFrame, SimulationResult
from Simulation_test_toolchain.pickle_compat import load_legacy_pandas_pickle
from Simulation_test_toolchain.policies.risk_idm import risk_idm_formula
from Simulation_test_toolchain.policies.ttc_utils import compute_ttc_simple


def load_scene_tracks(
    data_dir: Path,
    location: str,
    scene_id: str,
    cache: Optional[Dict[Tuple[str, str], Mapping[str, Any]]] = None,
) -> Mapping[str, Any]:
    key = (location, scene_id)
    if cache is not None and key in cache:
        return cache[key]
    path = data_dir / location / f"tp_info_{location}.pkl"
    with path.open("rb") as handle:
        tracks = load_legacy_pandas_pickle(handle)[scene_id]
    if cache is not None:
        cache[key] = tracks
    return tracks


def run_fast_risk_idm_simulation(
    cfg: ToolchainConfig,
    *,
    scene_tracks: Mapping[str, Any],
) -> SimulationResult:
    location = cfg.dataset.location
    scene_name = cfg.scenario.scene_name or f"{location}_{cfg.scenario.scene_index}"
    scene_id = _scene_id_from_name(location, scene_name)
    ego_id = str(cfg.scenario.ego_agent_name)
    init_timestep = int(cfg.scenario.init_timestep or 0)
    dt = float(cfg.dataset.desired_dt)
    requested_steps = int(cfg.scenario.num_steps)
    scene_length = _scene_length(scene_tracks)
    future_steps = int(np.ceil(cfg.simulation.future_sec / dt))
    window_end = min(scene_length, init_timestep + requested_steps + future_steps + 2)
    num_steps = max(0, min(requested_steps, window_end - init_timestep - 1))

    state_by_agent = _state_tables(scene_tracks)
    ego_table = state_by_agent.get(ego_id)
    if ego_table is None or init_timestep not in ego_table.index:
        raise ValueError(f"ego={ego_id!r} is not present at timestep={init_timestep}")

    active_agent_ids = [
        agent_id
        for agent_id, state in state_by_agent.items()
        if init_timestep in state.index
    ]
    if ego_id not in active_agent_ids:
        active_agent_ids.insert(0, ego_id)

    path = _reference_path(ego_table, init_timestep, init_timestep + num_steps + future_steps + 2)
    ego_initial = ego_table.loc[init_timestep]
    initial_speed = float(np.hypot(float(ego_initial["vx"]), float(ego_initial["vy"])))
    override = cfg.policies.ego.get("initial_velocity_override_mps")
    if override is not None and np.isfinite(float(override)):
        initial_speed = max(0.0, float(override))
    progress_m = float(path["arc"][0])
    ego_speed = initial_speed
    ego_observed_velocity = np.zeros(2, dtype=float)
    ego_x = float(ego_initial["x"])
    ego_y = float(ego_initial["y"])
    ego_heading = _heading_from_row(ego_initial)

    desired_velocity = float(cfg.policies.ego.get("desired_velocity", 15.0))
    max_acceleration = float(cfg.policies.ego.get("max_acceleration", 5.0))
    min_acceleration = float(cfg.policies.ego.get("min_acceleration", -5.0))
    neighbor_radius = float(cfg.policies.ego.get("neighbor_radius", cfg.simulation.neighbor_radius))
    inference_interval = max(1, int(cfg.policies.ego.get("inference_interval_steps", 5)))

    frames: List[AgentFrame] = []
    cached_acceleration = 0.0
    cached_ttc = 10.0
    cached_target_agent: Optional[str] = None
    has_cached_control = False
    action_step = 0

    _append_fast_frames(
        frames,
        timestep=init_timestep,
        active_agent_ids=active_agent_ids,
        state_by_agent=state_by_agent,
        ego_id=ego_id,
        ego_state=(ego_x, ego_y, ego_heading, float(np.linalg.norm(ego_observed_velocity))),
        ego_policy=cfg.policies.ego_policy,
        ego_command={},
    )

    for offset in range(1, num_steps + 1):
        current_timestep = init_timestep + offset - 1
        should_infer = (not has_cached_control) or (action_step % inference_interval == 0)
        if should_infer:
            ttc, collision_point, target_agent = _fast_min_ttc(
                ego_pos=np.array([ego_x, ego_y], dtype=float),
                ego_velocity=ego_observed_velocity,
                timestep=current_timestep,
                active_agent_ids=active_agent_ids,
                ego_id=ego_id,
                state_by_agent=state_by_agent,
                neighbor_radius=neighbor_radius,
            )
            se = (
                float(np.linalg.norm(collision_point - np.array([ego_x, ego_y], dtype=float)))
                if collision_point is not None
                else float("inf")
            )
            acceleration = risk_idm_formula(
                ve=ego_speed,
                se=se,
                r=ttc,
                v_0=desired_velocity,
                a_max=max_acceleration,
                a_min=min_acceleration,
            )
            cached_acceleration = float(acceleration)
            cached_ttc = float(ttc)
            cached_target_agent = target_agent
            has_cached_control = True
        else:
            acceleration = cached_acceleration
            ttc = cached_ttc
            target_agent = cached_target_agent

        ego_speed = max(0.0, ego_speed + float(acceleration) * dt)
        step_dist = ego_speed * dt
        progress_m += step_dist
        prev_x, prev_y = ego_x, ego_y
        ego_x, ego_y, ego_heading, path_exhausted = _interpolate_path(path, progress_m)
        ego_observed_velocity = np.array(
            [(ego_x - prev_x) / dt, (ego_y - prev_y) / dt],
            dtype=float,
        )
        action_step += 1
        command = {
            "policy": "risk_idm",
            "inference_interval_steps": inference_interval,
            "used_cached_control": not should_infer,
            "acceleration": float(acceleration),
            "velocity": float(ego_speed),
            "initial_velocity_override_mps": override,
            "min_ttc": float(ttc),
            "target_agent": target_agent,
            "reference_progress_m": float(step_dist),
            "reference_path_points": int(len(path["xy"])),
            "path_exhausted": bool(path_exhausted),
            "fast_risk_idm": True,
        }
        _append_fast_frames(
            frames,
            timestep=init_timestep + offset,
            active_agent_ids=active_agent_ids,
            state_by_agent=state_by_agent,
            ego_id=ego_id,
            ego_state=(ego_x, ego_y, ego_heading, float(np.linalg.norm(ego_observed_velocity))),
            ego_policy=cfg.policies.ego_policy,
            ego_command=command,
        )

    metadata = {
        "dataset": "sind",
        "location": location,
        "scene_index": cfg.scenario.scene_index,
        "scene_name": scene_name,
        "init_timestep": init_timestep,
        "window_end_timestep": window_end - 1,
        "num_steps": num_steps,
        "ego_agent": ego_id,
        "active_ego_agent": ego_id,
        "ego_policy": cfg.policies.ego_policy,
        "non_ego_policy": cfg.policies.non_ego_policy,
        "mode": cfg.simulation.mode,
        "controlled_neighbor_radius_m": cfg.simulation.controlled_neighbor_radius_m,
        "controlled_neighbor_max_agents": cfg.simulation.controlled_neighbor_max_agents,
        "controlled_neighbor_forward_only": cfg.simulation.controlled_neighbor_forward_only,
        "controlled_neighbor_min_speed_mps": cfg.simulation.controlled_neighbor_min_speed_mps,
        "cache_path": "",
        "map_name": f"sind:{location}",
        "fast_risk_idm": True,
    }
    if cfg.scenario.semantic_label_id:
        metadata["semantic_label_id"] = cfg.scenario.semantic_label_id
    if cfg.scenario.semantic_label:
        metadata["semantic_label"] = cfg.scenario.semantic_label
    return SimulationResult(metadata=metadata, frames=frames)


def _state_tables(scene_tracks: Mapping[str, Any]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for raw_id, tp_data in scene_tracks.items():
        state = tp_data.get("State")
        if state is None or state.empty or "frame_id" not in state.columns:
            continue
        table = state.copy().sort_values("frame_id").set_index("frame_id", drop=False)
        tables[str(raw_id)] = table
    return tables


def _reference_path(
    ego_table: pd.DataFrame,
    start_timestep: int,
    end_timestep: int,
) -> Dict[str, Any]:
    window = ego_table[
        (ego_table["frame_id"] >= start_timestep)
        & (ego_table["frame_id"] <= end_timestep)
    ].sort_index()
    xy = window[["x", "y"]].to_numpy(dtype=float)
    headings = np.unwrap(np.array([_heading_from_row(row) for _, row in window.iterrows()]))
    if len(xy) == 0:
        raise ValueError("empty ego reference path")
    if len(xy) == 1:
        arc = np.array([0.0], dtype=float)
    else:
        deltas = np.diff(xy, axis=0)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(deltas, axis=1))])
    return {"xy": xy, "heading": headings, "arc": arc}


def _interpolate_path(path: Mapping[str, Any], progress_m: float) -> Tuple[float, float, float, bool]:
    xy = path["xy"]
    headings = path["heading"]
    arc = path["arc"]
    if len(xy) == 1 or progress_m <= float(arc[0]):
        return float(xy[0, 0]), float(xy[0, 1]), _wrap_angle(float(headings[0])), False
    if progress_m >= float(arc[-1]):
        return float(xy[-1, 0]), float(xy[-1, 1]), _wrap_angle(float(headings[-1])), True
    idx = int(np.searchsorted(arc, progress_m, side="right") - 1)
    idx = max(0, min(idx, len(xy) - 2))
    denom = max(float(arc[idx + 1] - arc[idx]), 1e-6)
    ratio = float(np.clip((progress_m - float(arc[idx])) / denom, 0.0, 1.0))
    pos = xy[idx] + ratio * (xy[idx + 1] - xy[idx])
    heading = float(headings[idx] + ratio * (headings[idx + 1] - headings[idx]))
    return float(pos[0]), float(pos[1]), _wrap_angle(heading), False


def _append_fast_frames(
    frames: List[AgentFrame],
    *,
    timestep: int,
    active_agent_ids: List[str],
    state_by_agent: Mapping[str, pd.DataFrame],
    ego_id: str,
    ego_state: Tuple[float, float, float, float],
    ego_policy: str,
    ego_command: Dict[str, Any],
) -> None:
    ego_x, ego_y, ego_heading, ego_speed = ego_state
    for agent_id in active_agent_ids:
        is_ego = agent_id == ego_id
        if is_ego:
            row = state_by_agent[ego_id].loc[timestep] if timestep in state_by_agent[ego_id].index else None
            length, width = _extent_from_row(row)
            frames.append(
                AgentFrame(
                    timestep=int(timestep),
                    agent_name=ego_id,
                    agent_type=_agent_type_from_row(row),
                    x=float(ego_x),
                    y=float(ego_y),
                    heading=float(ego_heading),
                    speed=float(ego_speed),
                    length=length,
                    width=width,
                    is_ego=True,
                    policy=ego_policy,
                    command=dict(ego_command),
                )
            )
            continue
        table = state_by_agent.get(agent_id)
        if table is None or timestep not in table.index:
            continue
        row = table.loc[timestep]
        frames.append(_frame_from_row(timestep, agent_id, row))


def _frame_from_row(timestep: int, agent_id: str, row: Any) -> AgentFrame:
    length, width = _extent_from_row(row)
    return AgentFrame(
        timestep=int(timestep),
        agent_name=str(agent_id),
        agent_type=_agent_type_from_row(row),
        x=float(row["x"]),
        y=float(row["y"]),
        heading=_heading_from_row(row),
        speed=float(np.hypot(float(row["vx"]), float(row["vy"]))),
        length=length,
        width=width,
        is_ego=False,
        policy="ground_truth",
        command={},
    )


def _fast_min_ttc(
    *,
    ego_pos: np.ndarray,
    ego_velocity: np.ndarray,
    timestep: int,
    active_agent_ids: List[str],
    ego_id: str,
    state_by_agent: Mapping[str, pd.DataFrame],
    neighbor_radius: float,
) -> Tuple[float, Optional[np.ndarray], Optional[str]]:
    ego_vel = np.asarray(ego_velocity, dtype=float)
    best_ttc = 10.0
    best_point = None
    best_agent = None
    for agent_id in active_agent_ids:
        if agent_id == ego_id:
            continue
        table = state_by_agent.get(agent_id)
        if table is None or timestep not in table.index:
            continue
        row = table.loc[timestep]
        other_pos = np.array([float(row["x"]), float(row["y"])], dtype=float)
        if float(np.linalg.norm(other_pos - ego_pos)) > neighbor_radius:
            continue
        other_vel = np.array([float(row["vx"]), float(row["vy"])], dtype=float)
        ttc, point = compute_ttc_simple(ego_pos, ego_vel, other_pos, other_vel, max_ttc=10.0)
        if ttc < best_ttc:
            best_ttc = float(ttc)
            best_point = point
            best_agent = str(agent_id)
    return best_ttc, best_point, best_agent


def _scene_id_from_name(location: str, scene_name: str) -> str:
    prefix = f"{location}_"
    return scene_name[len(prefix) :] if scene_name.startswith(prefix) else scene_name


def _scene_length(scene_tracks: Mapping[str, Any]) -> int:
    max_frame = 0
    for tp_data in scene_tracks.values():
        state = tp_data.get("State")
        if state is not None and not state.empty and "frame_id" in state.columns:
            max_frame = max(max_frame, int(state["frame_id"].max()))
    return max_frame + 1


def _heading_from_row(row: Any) -> float:
    if row is None:
        return 0.0
    if "heading_rad" in row and np.isfinite(float(row["heading_rad"])):
        return float(row["heading_rad"])
    if "yaw_rad" in row and np.isfinite(float(row["yaw_rad"])):
        return float(row["yaw_rad"])
    return 0.0


def _extent_from_row(row: Any) -> Tuple[float, float]:
    if row is None:
        return 4.2, 1.8
    length = float(row["length"]) if "length" in row and np.isfinite(float(row["length"])) else 4.2
    width = float(row["width"]) if "width" in row and np.isfinite(float(row["width"])) else 1.8
    return max(length, 0.1), max(width, 0.1)


def _agent_type_from_row(row: Any) -> str:
    if row is None or "agent_type" not in row:
        return "UNKNOWN"
    return str(row["agent_type"])


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
