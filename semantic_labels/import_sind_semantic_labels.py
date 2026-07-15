from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from semantic_labels.labels import DEFAULT_LABEL_PATH, save_semantic_labels


DEFAULT_EXTRACT_ROOT = Path("datasets/SinD_Valued_Scenario_extract")
LOCATION_ALIASES = {"ccR": "cqR"}
SCENE_RE = re.compile(r"(?P<stem>.+?)(?:_event)?\.pkl$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import external SinD valued-scenario extracts as semantic labels."
    )
    parser.add_argument("--extract-root", default=str(DEFAULT_EXTRACT_ROOT))
    parser.add_argument("--data-dir", default="datasets/SinD_dataset")
    parser.add_argument("--output", default=str(DEFAULT_LABEL_PATH))
    parser.add_argument(
        "--mode",
        choices=("full", "sample"),
        default="full",
        help="Full imports all recognized labels. Sample keeps a small subset for quick inspection.",
    )
    parser.add_argument(
        "--max-narrow-parts-per-scene",
        type=int,
        default=None,
        help="Limit imported narrow-area part files per source scene.",
    )
    parser.add_argument(
        "--max-narrow-scenes-per-location",
        type=int,
        default=None,
        help="Limit imported narrow-area source scenes per location.",
    )
    args = parser.parse_args()

    extract_root = Path(args.extract_root)
    scene_index = build_scene_index(Path(args.data_dir))
    labels: List[Dict[str, Any]] = []
    labels.extend(import_mprttc(extract_root, scene_index))
    labels.extend(import_shielding(extract_root, scene_index))
    labels.extend(
        import_narrow_area(
            extract_root,
            scene_index,
            max_parts_per_scene=args.max_narrow_parts_per_scene,
            max_scenes_per_location=args.max_narrow_scenes_per_location,
            sample_mode=args.mode == "sample",
        )
    )
    labels = dedupe_and_sort(labels)
    save_semantic_labels(labels, args.output)

    counts = Counter(tag for label in labels for tag in label["semantic_tags"])
    print(f"Saved {len(labels)} labels to {args.output}")
    for tag, count in sorted(counts.items()):
        print(f"  {tag}: {count}")


def build_scene_index(data_dir: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    from collections import defaultdict

    from trajdata import AgentType, UnifiedDataset

    indexed: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    locations = sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_dir() and (p / f"frame_data_{p.name}.pkl").exists()
    )
    for location in locations:
        dataset = UnifiedDataset(
            desired_data=[f"sind-{location}"],
            data_dirs={"sind": str(data_dir)},
            only_types=[AgentType.VEHICLE],
            desired_dt=0.1,
            centric="agent",
            history_sec=(2.0, 2.0),
            future_sec=(4.0, 4.0),
            incl_raster_map=False,
            incl_vector_map=False,
            verbose=False,
            num_workers=0,
        )
        for idx, scene in enumerate(dataset.scenes()):
            indexed[location][scene.name] = {
                "scene_index": idx,
                "length_timesteps": int(scene.length_timesteps),
            }
    return dict(indexed)


def import_mprttc(
    extract_root: Path, scene_index: Dict[str, Dict[str, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    source_dir = extract_root / "MprTTC_sind_analysis" / "event_pkl"
    labels: List[Dict[str, Any]] = []
    if not source_dir.exists():
        return labels

    for pkl_path in sorted(source_dir.glob("*_event.pkl")):
        source_stem = parse_source_stem(pkl_path)
        location = "tj"
        scene_name = f"{location}_{source_stem}"
        scene_meta = scene_index.get(location, {}).get(scene_name)
        if scene_meta is None:
            continue
        data = load_pickle(pkl_path)
        for event_id, event in sorted(data.items(), key=lambda item: int(item[0])):
            start = int(event["start_frame"])
            end = int(event["end_frame"])
            ego_id, challenger_id = normalize_pair(event.get("TPID", []))
            labels.append(
                make_label(
                    scenario_id=f"SIND_TJ_MPRTTC_{source_slug(source_stem)}_{int(event_id):05d}",
                    location=location,
                    scene_name=scene_name,
                    scene_meta=scene_meta,
                    start_frame=start,
                    end_frame=end,
                    tags=["high_risk_mprttc"],
                    agents={"ego_id": ego_id, "challenger_id": challenger_id},
                    semantics={
                        "type": "mprttc",
                        "mprttc_values": event.get("mprttc", []),
                        "min_mprttc": min(event.get("mprttc", []) or [None]),
                    },
                    source={"path": str(pkl_path), "event_id": event_id},
                )
            )
    return labels


def import_shielding(
    extract_root: Path, scene_index: Dict[str, Dict[str, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    source_root = extract_root / "Sind_Visual_Shielding"
    labels: List[Dict[str, Any]] = []
    if not source_root.exists():
        return labels

    for pkl_path in sorted(source_root.glob("*/*.pkl")):
        if pkl_path.name.startswith("tp_info_"):
            continue
        if pkl_path.name.startswith("shielding_scenarios_"):
            source_stem = None
        else:
            source_stem = parse_source_stem(pkl_path)
        location = normalize_location(pkl_path.parent.name)
        data = load_pickle(pkl_path)
        if not isinstance(data, dict):
            continue
        for key, event in sorted(data.items(), key=lambda item: str(item[0])):
            frame_window = event.get("shielding_frame_ids")
            if not frame_window or len(frame_window) != 2:
                continue
            scene_name = choose_scene_name(location, source_stem, event, scene_index)
            scene_meta = scene_index.get(location, {}).get(scene_name)
            if scene_meta is None:
                continue
            main_id, shielding_id, shielded_id = normalize_triplet(key)
            start, end = int(frame_window[0]), int(frame_window[1])
            labels.append(
                make_label(
                    scenario_id=(
                        f"SIND_{location.upper()}_SHIELD_"
                        f"{source_slug(scene_name)}_{source_slug(str(key))}"
                    ),
                    location=location,
                    scene_name=scene_name,
                    scene_meta=scene_meta,
                    start_frame=start,
                    end_frame=end,
                    tags=["visual_shielding"],
                    agents={
                        "ego_id": main_id,
                        "shielding_id": shielding_id,
                        "shielded_id": shielded_id,
                    },
                    semantics={
                        "type": "visual_shielding",
                        "conflict_type": event.get("Type of conflict")
                        or event.get("type_of_conflict"),
                        "shielding_frame_length": event.get("shielding_frame_length"),
                        "main_vehicle_type": event.get("main_vehicle_type"),
                        "shielding_vehicle_type": event.get("shielding_vehicle_type"),
                        "shielded_vehicle_type": event.get("shielded_vehicle_type"),
                        "time_difference": event.get("time_difference"),
                        "time_to_intersection": event.get("time_to_intersection"),
                    },
                    source={"path": str(pkl_path), "event_key": repr(key)},
                )
            )
    return labels


def import_narrow_area(
    extract_root: Path,
    scene_index: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    max_parts_per_scene: Optional[int],
    max_scenes_per_location: Optional[int],
    sample_mode: bool,
) -> List[Dict[str, Any]]:
    source_root = extract_root / "SinD_availabel_area"
    labels: List[Dict[str, Any]] = []
    if not source_root.exists():
        return labels

    for location_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        location = normalize_location(location_dir.name)
        scenario_root = location_dir / "scenarios"
        if not scenario_root.exists():
            continue
        imported_scenes = 0
        for scene_dir in sorted(p for p in scenario_root.iterdir() if p.is_dir()):
            if sample_mode and max_scenes_per_location is not None and imported_scenes >= max_scenes_per_location:
                break
            scene_stem = scene_dir.name
            scene_name = f"{location}_{scene_stem}"
            scene_meta = scene_index.get(location, {}).get(scene_name)
            if scene_meta is None:
                continue
            part_files = sorted(
                scene_dir.glob("*.pkl"),
                key=lambda p: (p.stat().st_size, natural_key(p)),
            )
            imported = 0
            for part_path in part_files:
                if sample_mode and max_parts_per_scene is not None and imported >= max_parts_per_scene:
                    break
                data = load_pickle(part_path)
                for ego_id, frames in sorted(data.items(), key=lambda item: str(item[0])):
                    if not frames:
                        continue
                    frame_ids = sorted(int(frame["frame_id"]) for frame in frames)
                    risk_frames = [frame for frame in frames if "max_area" in frame]
                    max_areas = [
                        int(frame["max_area"])
                        for frame in risk_frames
                        if frame.get("max_area") is not None
                    ]
                    start, end = frame_ids[0], frame_ids[-1]
                    labels.append(
                        make_label(
                            scenario_id=(
                                f"SIND_{location.upper()}_NARROW_"
                                f"{source_slug(scene_stem)}_{source_slug(part_path.stem)}_"
                                f"EGO{source_slug(str(ego_id))}"
                            ),
                            location=location,
                            scene_name=scene_name,
                            scene_meta=scene_meta,
                            start_frame=start,
                            end_frame=end,
                            tags=["narrow_feasible_area"],
                            agents={"ego_id": ego_id},
                            semantics={
                                "type": "narrow_feasible_area",
                                "min_max_area": min(max_areas) if max_areas else None,
                                "max_area_values": max_areas[:200],
                                "risk_frame_count": len(risk_frames),
                            },
                            source={"path": str(part_path), "ego_id": ego_id},
                        )
                    )
                imported += 1
            if imported:
                imported_scenes += 1
    return labels


def make_label(
    *,
    scenario_id: str,
    location: str,
    scene_name: str,
    scene_meta: Dict[str, Any],
    start_frame: int,
    end_frame: int,
    tags: List[str],
    agents: Dict[str, Any],
    semantics: Dict[str, Any],
    source: Dict[str, Any],
) -> Dict[str, Any]:
    start = max(0, int(start_frame))
    end = max(start, int(end_frame))
    scene_len = int(scene_meta["length_timesteps"])
    if start >= scene_len:
        start = max(0, scene_len - 1)
    if end >= scene_len:
        end = scene_len - 1
    return {
        "scenario_id": scenario_id,
        "dataset": "sind",
        "location": location,
        "source_scene_name": scene_name,
        "scene_index": int(scene_meta["scene_index"]),
        "time_window": {
            "start_frame": start,
            "end_frame": end,
            "duration_sec": round((end - start + 1) * 0.1, 3),
        },
        "agents": stringify_agents(agents),
        "semantic_tags": tags,
        "semantics": clean_json(semantics),
        "source": clean_json(source),
    }


def choose_scene_name(
    location: str,
    source_stem: Optional[str],
    event: Dict[str, Any],
    scene_index: Dict[str, Dict[str, Dict[str, Any]]],
) -> str:
    if source_stem:
        return f"{location}_{source_stem}"
    frame_window = event.get("shielding_frame_ids") or (0, 0)
    start = int(frame_window[0])
    candidates = scene_index.get(location, {})
    for name, meta in candidates.items():
        if 0 <= start < int(meta["length_timesteps"]):
            return name
    return next(iter(candidates), f"{location}_unknown")


def parse_source_stem(path: Path) -> str:
    match = SCENE_RE.match(path.name)
    stem = match.group("stem") if match else path.stem
    if stem.startswith("xian_"):
        return stem
    return stem


def normalize_location(value: str) -> str:
    return LOCATION_ALIASES.get(value, value)


def normalize_pair(values: Iterable[Any]) -> Tuple[str, Optional[str]]:
    items = list(values)
    ego = items[0] if items else None
    other = items[1] if len(items) > 1 else None
    return str(ego), None if other is None else str(other)


def normalize_triplet(value: Any) -> Tuple[str, Optional[str], Optional[str]]:
    if isinstance(value, tuple) and len(value) >= 3:
        return str(value[0]), str(value[1]), str(value[2])
    return str(value), None, None


def stringify_agents(agents: Dict[str, Any]) -> Dict[str, Any]:
    return {key: None if value is None else str(value) for key, value in agents.items()}


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def source_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    return slug or "UNKNOWN"


def natural_key(path: Path) -> List[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def dedupe_and_sort(labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for label in labels:
        scenario_id = label["scenario_id"]
        if scenario_id in seen:
            continue
        seen.add(scenario_id)
        deduped.append(label)
    return sorted(
        deduped,
        key=lambda item: (
            item["location"],
            item["source_scene_name"],
            item["time_window"]["start_frame"],
            item["scenario_id"],
        ),
    )


if __name__ == "__main__":
    main()
