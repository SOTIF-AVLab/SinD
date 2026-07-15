from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .config import ToolchainConfig
from .ego_selection import select_ego_from_scene
from .records import AgentFrame, SimulationResult
from .state_utils import (
    get_agent_world_pose,
    get_ground_truth_next_xyh,
    xyh_to_state_array,
)
from Simulation_test_toolchain.policies.registry import build_policy


@dataclass
class LoadedSimulationDataset:
    dataset: Any
    scenes: List[Any]


def load_simulation_dataset(cfg: ToolchainConfig) -> LoadedSimulationDataset:
    from trajdata import AgentType, UnifiedDataset

    print(
        f"[toolchain] loading {cfg.dataset.name}-{cfg.dataset.location} "
        f"scene_index={cfg.scenario.scene_index}",
        flush=True,
    )

    ego_policy_name, non_ego_policy_name = _policy_names(cfg)
    needs_vector_map, use_raster_map = _dataset_requirements(
        cfg, ego_policy_name, non_ego_policy_name
    )
    only_types = _agent_types_from_config(AgentType, cfg.dataset.agent_types)
    dataset = UnifiedDataset(
        desired_data=[f"sind-{cfg.dataset.location}"],
        data_dirs={"sind": str(Path(cfg.dataset.data_dir))},
        only_types=only_types,
        agent_interaction_distances=defaultdict(
            lambda: float(cfg.simulation.neighbor_radius)
        ),
        desired_dt=cfg.dataset.desired_dt,
        centric="agent",
        history_sec=(cfg.simulation.history_sec, cfg.simulation.history_sec),
        future_sec=(cfg.simulation.future_sec, cfg.simulation.future_sec),
        incl_raster_map=use_raster_map,
        raster_map_params=(
            {
                "px_per_m": 12,
                "map_size_px": 224,
                "offset_frac_xy": (-0.5, 0.0),
                "use_lanelet2_maps": cfg.dataset.use_lanelet2_maps,
            }
            if use_raster_map
            else None
        ),
        incl_vector_map=needs_vector_map,
        vector_map_params={
            "collate": needs_vector_map,
            "associate_traffic_lights": False,
            "incl_road_lanes": True,
            "incl_road_areas": True,
            "incl_ped_crosswalks": True,
            "incl_ped_walkways": True,
        },
        verbose=True,
        num_workers=0,
    )
    scenes = list(dataset.scenes())
    print(
        f"[toolchain] dataset ready: scenes={len(scenes)}, samples={len(dataset)}",
        flush=True,
    )
    return LoadedSimulationDataset(dataset=dataset, scenes=scenes)


def run_simulation(
    cfg: ToolchainConfig,
    dataset_bundle: Optional[LoadedSimulationDataset] = None,
    policy_cache: Optional[Dict[Tuple[Any, ...], Optional[Any]]] = None,
) -> SimulationResult:
    from trajdata.simulation import SimulationScene

    ego_policy_name, non_ego_policy_name = _policy_names(cfg)
    if dataset_bundle is None:
        dataset_bundle = load_simulation_dataset(cfg)
    dataset = dataset_bundle.dataset
    scenes = dataset_bundle.scenes

    if cfg.scenario.scene_name:
        scene = next(
            (candidate for candidate in scenes if candidate.name == cfg.scenario.scene_name),
            None,
        )
        if scene is None:
            available = ", ".join(candidate.name for candidate in scenes[:5])
            raise ValueError(
                f"scenario.scene_name={cfg.scenario.scene_name!r} not found. "
                f"First available scenes: {available}"
            )
    else:
        scene = scenes[cfg.scenario.scene_index]
    ego_agent, _ = select_ego_from_scene(
        scene,
        strategy=cfg.scenario.ego_selection_strategy,
        ego_agent_name=cfg.scenario.ego_agent_name,
    )
    requested_ego_name = ego_agent.name
    init_timestep = cfg.scenario.init_timestep
    if init_timestep is None:
        init_timestep = min(
            int(cfg.simulation.history_sec / cfg.dataset.desired_dt) + 10,
            max(0, scene.length_timesteps // 2),
        )
    windowed_scene = _make_windowed_scene(scene, init_timestep, cfg)

    sim_scene = SimulationScene(
        env_name=f"sind_{cfg.dataset.location}_test",
        scene_name=f"scene_{cfg.scenario.scene_index:03d}_test",
        scene=windowed_scene,
        dataset=dataset,
        init_timestep=init_timestep,
        freeze_agents=True,
    )
    _ensure_sim_cache_defaults(sim_scene)
    obs = sim_scene.reset()
    ego_idx = _find_agent_idx(obs.agent_name, ego_agent.name)
    if ego_idx is None:
        if not cfg.scenario.allow_ego_fallback:
            raise ValueError(
                f"Requested ego '{requested_ego_name}' is not present after dataset "
                f"agent filtering at timestep {init_timestep}."
            )
        ego_idx = 0
        active_ego_name = str(obs.agent_name[ego_idx])
        print(
            f"[toolchain] requested ego '{requested_ego_name}' not present; "
            f"falling back to present agent '{active_ego_name}' at timestep {init_timestep}",
            flush=True,
        )
    else:
        active_ego_name = ego_agent.name
    print(
        f"[toolchain] simulation reset: scene={scene.name}, init_timestep={init_timestep}, "
        f"agents={len(obs.agent_name)}, ego={active_ego_name}",
        flush=True,
    )

    ego_policy = _policy_for_config(
        ego_policy_name,
        cfg=cfg,
        params=cfg.policies.ego,
        cache=policy_cache,
    )
    non_ego_policy = _policy_for_config(
        non_ego_policy_name,
        cfg=cfg,
        params=cfg.policies.non_ego,
        cache=policy_cache,
    )
    if ego_policy is not None:
        ego_policy.reset(obs, ego_idx)
    if non_ego_policy is not None:
        non_ego_policy.reset(obs, ego_idx)

    frames: List[AgentFrame] = []
    _append_frames(
        frames,
        obs,
        ego_idx,
        init_timestep,
        ego_policy_name,
        {},
        non_ego_policy_name,
        set(),
        {},
    )

    max_available = max(0, windowed_scene.length_timesteps - init_timestep - 1)
    num_steps = min(cfg.scenario.num_steps, max_available)
    print(
        f"[toolchain] stepping: requested={cfg.scenario.num_steps}, actual={num_steps}, "
        f"ego={active_ego_name}, policy={ego_policy_name}",
        flush=True,
    )
    for offset in range(1, num_steps + 1):
        current_ego_idx = _find_agent_idx(obs.agent_name, active_ego_name)
        if current_ego_idx is None:
            break

        controlled_non_ego = _select_controlled_non_ego_indices(
            obs, current_ego_idx, cfg
        )
        controlled_non_ego_names = {
            str(obs.agent_name[idx])
            for idx in controlled_non_ego
            if idx < len(obs.agent_name)
        }
        actions = {}
        ego_command = {}
        non_ego_commands: Dict[str, Dict] = {}
        for idx, agent_name in enumerate(obs.agent_name):
            if idx == current_ego_idx and ego_policy is not None:
                action = ego_policy.get_action(obs, idx)
                xyh = action.xyh
                ego_command = action.command
                if not np.isfinite(xyh).all():
                    xyh = get_ground_truth_next_xyh(obs, idx)
                    ego_command = {
                        **ego_command,
                        "fallback": "ground_truth",
                        "fallback_reason": "policy returned non-finite xyh",
                    }
            elif idx in controlled_non_ego and non_ego_policy is not None:
                action = non_ego_policy.get_action(obs, idx)
                xyh = action.xyh
                non_ego_commands[str(agent_name)] = action.command
                if not np.isfinite(xyh).all():
                    xyh = get_ground_truth_next_xyh(obs, idx)
                    non_ego_commands[str(agent_name)] = {
                        **action.command,
                        "fallback": "ground_truth",
                        "fallback_reason": "policy returned non-finite xyh",
                    }
            else:
                xyh = get_ground_truth_next_xyh(obs, idx)
            actions[agent_name] = xyh_to_state_array(xyh)

        obs = sim_scene.step(actions)
        next_ego_idx = _find_agent_idx(obs.agent_name, active_ego_name)
        if next_ego_idx is None:
            break
        _append_frames(
            frames,
            obs,
            next_ego_idx,
            init_timestep + offset,
            ego_policy_name,
            ego_command,
            non_ego_policy_name,
            controlled_non_ego_names,
            non_ego_commands,
        )

    print(f"[toolchain] collected frames={len(frames)}", flush=True)

    metadata = {
        "dataset": "sind",
        "location": cfg.dataset.location,
        "scene_index": cfg.scenario.scene_index,
        "scene_name": scene.name,
        "init_timestep": init_timestep,
        "window_end_timestep": windowed_scene.length_timesteps - 1,
        "num_steps": num_steps,
        "ego_agent": requested_ego_name,
        "active_ego_agent": active_ego_name,
        "ego_policy": ego_policy_name,
        "non_ego_policy": cfg.policies.non_ego_policy,
        "mode": cfg.simulation.mode,
        "controlled_neighbor_radius_m": cfg.simulation.controlled_neighbor_radius_m,
        "controlled_neighbor_max_agents": cfg.simulation.controlled_neighbor_max_agents,
        "controlled_neighbor_forward_only": cfg.simulation.controlled_neighbor_forward_only,
        "controlled_neighbor_min_speed_mps": cfg.simulation.controlled_neighbor_min_speed_mps,
        "cache_path": str(dataset.cache_path),
        "map_name": f"sind:{cfg.dataset.location}",
    }
    if cfg.scenario.semantic_label_id:
        metadata["semantic_label_id"] = cfg.scenario.semantic_label_id
    if cfg.scenario.semantic_label:
        metadata["semantic_label"] = cfg.scenario.semantic_label
    if getattr(obs, "map_names", None) is not None and len(obs.map_names) > 0:
        metadata["map_name"] = obs.map_names[next_ego_idx]
    return SimulationResult(metadata=metadata, frames=frames)


def _policy_names(cfg: ToolchainConfig) -> Tuple[str, str]:
    if cfg.simulation.mode == "open_loop":
        ego_policy_name = "ground_truth"
    else:
        ego_policy_name = cfg.policies.ego_policy

    non_ego_policy_name = (
        cfg.policies.non_ego_policy
        if cfg.simulation.mode == "multi_agent_closed_loop"
        else "ground_truth"
    )
    return ego_policy_name, non_ego_policy_name


def _dataset_requirements(
    cfg: ToolchainConfig,
    ego_policy_name: str,
    non_ego_policy_name: str,
) -> Tuple[bool, bool]:
    needs_vector_map = False
    use_raster_map = bool(
        cfg.policies.ego.get("require_raster_map", False)
        or cfg.policies.non_ego.get("require_raster_map", False)
    )
    return needs_vector_map, use_raster_map


def _agent_types_from_config(agent_type_cls, names: Optional[List[str]]):
    if names is None:
        return [agent_type_cls.VEHICLE]
    if len(names) == 0:
        return None

    resolved = []
    for raw_name in names:
        name = str(raw_name).strip()
        if not name:
            continue
        try:
            resolved.append(getattr(agent_type_cls, name.upper()))
        except AttributeError as exc:
            valid = ", ".join(member.name for member in agent_type_cls)
            raise ValueError(
                f"Unknown dataset.agent_types entry {raw_name!r}. Valid values: {valid}"
            ) from exc
    return resolved or None


def _policy_for_config(
    policy_name: str,
    *,
    cfg: ToolchainConfig,
    params: Dict[str, Any],
    cache: Optional[Dict[Tuple[Any, ...], Optional[Any]]],
):
    params_for_cache = dict(params)
    params_for_cache.pop("initial_velocity_override_mps", None)
    if cache is None or policy_name == "ground_truth":
        return build_policy(
            policy_name,
            dt=cfg.dataset.desired_dt,
            params=params,
            checkpoints=cfg.checkpoints,
        )

    key = (
        policy_name,
        float(cfg.dataset.desired_dt),
        _freeze_for_cache(params_for_cache),
        _freeze_for_cache(cfg.checkpoints),
    )
    if key not in cache:
        cache[key] = build_policy(
            policy_name,
            dt=cfg.dataset.desired_dt,
            params=params,
            checkpoints=cfg.checkpoints,
        )
    policy = cache[key]
    if policy is not None and hasattr(policy, "initial_velocity_override_mps"):
        setattr(policy, "initial_velocity_override_mps", params.get("initial_velocity_override_mps"))
    return policy


def _freeze_for_cache(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return tuple(
            (field, _freeze_for_cache(getattr(value, field)))
            for field in sorted(value.__dataclass_fields__)
        )
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_for_cache(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_cache(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_for_cache(item) for item in value))
    return value


def _make_windowed_scene(scene, init_timestep: int, cfg: ToolchainConfig):
    from trajdata.data_structures.scene_metadata import Scene

    future_steps = int(np.ceil(cfg.simulation.future_sec / cfg.dataset.desired_dt))
    window_end = min(
        scene.length_timesteps,
        init_timestep + cfg.scenario.num_steps + future_steps + 2,
    )
    agent_presence = scene.agent_presence[:window_end]
    agent_names = {
        agent.name for present_agents in agent_presence for agent in present_agents
    }
    agents = [agent for agent in scene.agents if agent.name in agent_names]

    return Scene(
        env_metadata=scene.env_metadata,
        name=scene.name,
        location=scene.location,
        data_split=scene.data_split,
        length_timesteps=window_end,
        raw_data_idx=scene.raw_data_idx,
        data_access_info=scene.data_access_info,
        description=scene.description,
        agents=agents,
        agent_presence=agent_presence,
    )


def _find_agent_idx(agent_names, target: str) -> Optional[int]:
    for idx, name in enumerate(agent_names):
        if name == target:
            return idx
    return None


def _select_controlled_non_ego_indices(
    obs,
    ego_idx: int,
    cfg: ToolchainConfig,
) -> Set[int]:
    if cfg.simulation.mode != "multi_agent_closed_loop":
        return set()

    max_agents = max(0, int(cfg.simulation.controlled_neighbor_max_agents))
    if max_agents <= 0:
        return set()

    ego_pose = get_agent_world_pose(obs, ego_idx)
    ego_pos = np.asarray(ego_pose["position"], dtype=float)
    ego_heading = float(ego_pose["heading"])
    ego_forward = np.array([np.cos(ego_heading), np.sin(ego_heading)], dtype=float)
    radius = float(cfg.simulation.controlled_neighbor_radius_m)
    min_speed = float(cfg.simulation.controlled_neighbor_min_speed_mps)
    candidates = []

    for idx in range(len(obs.agent_name)):
        if idx == ego_idx:
            continue
        pose = get_agent_world_pose(obs, idx)
        rel = np.asarray(pose["position"], dtype=float) - ego_pos
        dist = float(np.linalg.norm(rel))
        if not np.isfinite(dist) or dist > radius:
            continue
        forward_dist = float(np.dot(rel, ego_forward))
        if cfg.simulation.controlled_neighbor_forward_only and forward_dist <= 0.0:
            continue
        speed = float(np.linalg.norm(pose["velocity"]))
        if not np.isfinite(speed) or speed < min_speed:
            continue
        candidates.append((forward_dist, dist, idx))

    if cfg.simulation.controlled_neighbor_forward_only:
        candidates.sort(key=lambda item: (item[0], item[1]))
    else:
        candidates.sort(key=lambda item: item[1])
    return {idx for _, _, idx in candidates[:max_agents]}


def _ensure_sim_cache_defaults(sim_scene) -> None:
    # SimulationDataFrameCache.reset() does not initialize these optional
    # transform attributes, but append_state()->get_value() expects them.
    for attr in ("_transf_mean", "_transf_rotmat"):
        if not hasattr(sim_scene.cache, attr):
            setattr(sim_scene.cache, attr, None)


def _append_frames(
    frames: List[AgentFrame],
    obs,
    ego_idx: int,
    scene_timestep: int,
    ego_policy_name: str,
    ego_command: Dict,
    non_ego_policy_name: str,
    controlled_non_ego_names: Set[str],
    non_ego_commands: Dict[str, Dict],
) -> None:
    for idx, name in enumerate(obs.agent_name):
        pose = get_agent_world_pose(obs, idx)
        speed = float(np.linalg.norm(pose["velocity"]))
        extent = pose["extent"]
        agent_type = "UNKNOWN"
        if getattr(obs, "agent_type", None) is not None:
            agent_type = str(obs.agent_type[idx].item())
        is_ego = idx == ego_idx
        agent_name = str(name)
        is_controlled_non_ego = agent_name in controlled_non_ego_names
        frames.append(
            AgentFrame(
                timestep=int(scene_timestep),
                agent_name=agent_name,
                agent_type=agent_type,
                x=float(pose["position"][0]),
                y=float(pose["position"][1]),
                heading=float(pose["heading"]),
                speed=speed,
                length=float(extent[0]) if len(extent) > 0 else 4.2,
                width=float(extent[1]) if len(extent) > 1 else 1.8,
                is_ego=is_ego,
                policy=(
                    ego_policy_name
                    if is_ego
                    else non_ego_policy_name
                    if is_controlled_non_ego
                    else "ground_truth"
                ),
                command=(
                    ego_command
                    if is_ego
                    else non_ego_commands.get(agent_name, {})
                    if is_controlled_non_ego
                    else {}
                ),
            )
        )
