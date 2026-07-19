from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_LABEL_PATH = Path("datasets/SinD_dataset/Semantic_labels/scenarios.json")


class SemanticLabelError(ValueError):
    """Raised when semantic labels are missing or inconsistent."""


def load_semantic_labels(path: str | Path = DEFAULT_LABEL_PATH) -> List[Dict[str, Any]]:
    label_path = Path(path)
    if not label_path.exists():
        raise SemanticLabelError(f"Semantic label file not found: {label_path}")
    with label_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    labels = raw.get("scenarios", raw) if isinstance(raw, dict) else raw
    if not isinstance(labels, list):
        raise SemanticLabelError(f"Semantic labels must be a list: {label_path}")
    return labels


def save_semantic_labels(
    labels: Iterable[Dict[str, Any]], path: str | Path = DEFAULT_LABEL_PATH
) -> None:
    label_path = Path(path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "dataset": "sind",
        "scenarios": list(labels),
    }
    with label_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def find_label(
    scenario_id: str, path: str | Path = DEFAULT_LABEL_PATH
) -> Dict[str, Any]:
    for label in load_semantic_labels(path):
        if label.get("scenario_id") == scenario_id:
            return label
    raise SemanticLabelError(f"Semantic scenario_id not found: {scenario_id}")


def query_semantic_labels(
    labels: Optional[Iterable[Dict[str, Any]]] = None,
    *,
    path: str | Path = DEFAULT_LABEL_PATH,
    location: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    min_duration: Optional[float] = None,
    ego_id: Optional[str | int] = None,
) -> List[Dict[str, Any]]:
    source = list(labels) if labels is not None else load_semantic_labels(path)
    tag_set = set(tags or [])
    ego_name = None if ego_id is None else str(ego_id)
    matched: List[Dict[str, Any]] = []
    for label in source:
        if location is not None and label.get("location") != location:
            continue
        if tag_set and not tag_set.issubset(set(label.get("semantic_tags", []))):
            continue
        if min_duration is not None:
            duration = float(label.get("time_window", {}).get("duration_sec", 0.0))
            if duration < min_duration:
                continue
        if ego_name is not None:
            label_ego = label.get("agents", {}).get("ego_id")
            if str(label_ego) != ego_name:
                continue
        matched.append(label)
    return matched


def resolve_label_for_toolchain(
    cfg: Any,
    label: Dict[str, Any],
    *,
    min_num_steps: int = 150,
) -> Any:
    """Return a copy of a toolchain config with scenario fields resolved."""

    resolved = copy.deepcopy(cfg)
    if label.get("dataset") and label.get("dataset") != "sind":
        raise SemanticLabelError(f"Unsupported label dataset: {label.get('dataset')}")

    resolved.dataset.name = "sind"
    resolved.dataset.location = label["location"]
    resolved.scenario.scene_index = int(label["scene_index"])
    resolved.scenario.init_timestep = int(label["time_window"]["start_frame"])
    resolved.scenario.ego_agent_name = str(label["agents"]["ego_id"])
    label_steps = (
        int(label["time_window"]["end_frame"])
        - int(label["time_window"]["start_frame"])
        + 1
    )
    resolved.scenario.num_steps = max(
        int(resolved.scenario.num_steps), int(label_steps), int(min_num_steps)
    )

    resolved.scenario.semantic_label_id = label["scenario_id"]
    resolved.scenario.semantic_label = label
    return resolved
