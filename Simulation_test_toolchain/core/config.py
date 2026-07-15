from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DatasetConfig:
    name: str = "sind"
    location: str = "tj"
    data_dir: str = "datasets/SinD_dataset"
    desired_dt: float = 0.1
    use_lanelet2_maps: bool = True
    agent_types: Optional[List[str]] = None


@dataclass
class ScenarioConfig:
    scene_index: int = 0
    scene_name: Optional[str] = None
    init_timestep: Optional[int] = None
    num_steps: int = 50
    ego_selection_strategy: str = "longest_trajectory"
    ego_agent_name: Optional[str] = None
    semantic_label_id: Optional[str] = None
    semantic_label_path: str = "datasets/SinD_dataset/Semantic_labels/scenarios.json"
    semantic_min_num_steps: int = 150
    semantic_label: Optional[Dict[str, Any]] = None
    allow_ego_fallback: bool = True


@dataclass
class SimulationConfig:
    mode: str = "ego_closed_loop"
    history_sec: float = 2.0
    future_sec: float = 4.0
    neighbor_radius: float = 50.0
    controlled_neighbor_radius_m: float = 25.0
    controlled_neighbor_max_agents: int = 5
    controlled_neighbor_forward_only: bool = True
    controlled_neighbor_min_speed_mps: float = 0.3


@dataclass
class PolicyConfig:
    ego_policy: str = "risk_idm"
    non_ego_policy: str = "ground_truth"
    ego: Dict[str, Any] = field(default_factory=dict)
    non_ego: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    diffuser_ckpt_path: Optional[str] = None


@dataclass
class OutputConfig:
    project_name: str = "sind_single_scene"
    root_dir: str = "Simulation_test_toolchain/test_projects"
    save_html: bool = True
    save_json: bool = True
    save_csv: bool = True


@dataclass
class VisualizationConfig:
    enabled: bool = True
    map_margin_m: float = 15.0
    width: int = 1200
    height: int = 760


@dataclass
class ToolchainConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    policies: PolicyConfig = field(default_factory=PolicyConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def _build_dataclass(cls, data: Dict[str, Any]):
    allowed = {field.name for field in cls.__dataclass_fields__.values()}
    return cls(**{key: value for key, value in data.items() if key in allowed})


def load_config(path: str | Path) -> ToolchainConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = ToolchainConfig(
        dataset=_build_dataclass(DatasetConfig, raw.get("dataset", {})),
        scenario=_build_dataclass(ScenarioConfig, raw.get("scenario", {})),
        simulation=_build_dataclass(SimulationConfig, raw.get("simulation", {})),
        policies=_build_dataclass(PolicyConfig, raw.get("policies", {})),
        checkpoints=_build_dataclass(CheckpointConfig, raw.get("checkpoints", {})),
        output=_build_dataclass(OutputConfig, raw.get("output", {})),
        visualization=_build_dataclass(
            VisualizationConfig, raw.get("visualization", {})
        ),
    )
    cfg = resolve_semantic_scenario(cfg)
    validate_config(cfg)
    return cfg


def resolve_semantic_scenario(cfg: ToolchainConfig) -> ToolchainConfig:
    if not cfg.scenario.semantic_label_id:
        return cfg

    from semantic_labels import find_label, resolve_label_for_toolchain

    label = find_label(cfg.scenario.semantic_label_id, cfg.scenario.semantic_label_path)
    return resolve_label_for_toolchain(
        cfg,
        label,
        min_num_steps=cfg.scenario.semantic_min_num_steps,
    )


def validate_config(cfg: ToolchainConfig) -> None:
    if cfg.dataset.name.lower() != "sind":
        raise ValueError("First version supports dataset.name='sind' only.")

    valid_modes = {"open_loop", "ego_closed_loop", "multi_agent_closed_loop"}
    if cfg.simulation.mode not in valid_modes:
        raise ValueError(f"simulation.mode must be one of {sorted(valid_modes)}.")

    valid_ego = {"ground_truth", "risk_idm", "diffuser"}
    if cfg.policies.ego_policy not in valid_ego:
        raise ValueError(f"policies.ego_policy must be one of {sorted(valid_ego)}.")

    valid_non_ego = {"ground_truth", "diffuser"}
    if cfg.policies.non_ego_policy not in valid_non_ego:
        raise ValueError(
            f"policies.non_ego_policy must be one of {sorted(valid_non_ego)}."
        )

    if cfg.scenario.num_steps <= 0:
        raise ValueError("scenario.num_steps must be positive.")

    if cfg.scenario.semantic_min_num_steps <= 0:
        raise ValueError("scenario.semantic_min_num_steps must be positive.")


def make_project_dir(cfg: ToolchainConfig, config_path: str | Path) -> Path:
    root = Path(cfg.output.root_dir)
    project_dir = root / cfg.output.project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    copied_config = project_dir / "config.yaml"
    source_text = Path(config_path).read_text(encoding="utf-8")
    copied_config.write_text(source_text, encoding="utf-8")
    return project_dir
