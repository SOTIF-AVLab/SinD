#!/usr/bin/env python3
"""Import offset-corrected SinD red-light violations as semantic labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic_labels.import_sind_semantic_labels import (  # noqa: E402
    build_scene_index,
    clean_json,
    source_slug,
)
from semantic_labels.labels import save_semantic_labels  # noqa: E402


DEFAULT_INPUT = Path("risk_mining/output_red_light_offset_corrected/red_light_violations.csv")
DEFAULT_DATA_DIR = Path("datasets/SinD_dataset")
DEFAULT_OUTPUT = Path(
    "datasets/SinD_dataset/Semantic_labels/scenarios_with_offset_redlight.json"
)
DEFAULT_SKIPPED_OUTPUT = Path(
    "risk_mining/output_red_light_offset_corrected/semantic_label_import_skipped.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skipped-output", type=Path, default=DEFAULT_SKIPPED_OUTPUT)
    parser.add_argument(
        "--window-before-frames",
        type=int,
        default=20,
        help="Frames to include before first stop-line contact.",
    )
    parser.add_argument(
        "--window-after-frames",
        type=int,
        default=20,
        help="Frames to include after the after-10m key frame.",
    )
    args = parser.parse_args()

    labels, skipped = import_labels(
        input_csv=args.input,
        data_dir=args.data_dir,
        window_before_frames=args.window_before_frames,
        window_after_frames=args.window_after_frames,
    )
    save_semantic_labels(labels, args.output)
    write_skipped_rows(skipped, args.skipped_output)
    print(f"Saved {len(labels)} offset-corrected red-light labels to {args.output}")
    print(f"Saved {len(skipped)} skipped rows to {args.skipped_output}")


def import_labels(
    *,
    input_csv: Path,
    data_dir: Path,
    window_before_frames: int,
    window_after_frames: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = read_csv(input_csv)
    scene_index = build_scene_index(data_dir)
    labels: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for row in rows:
        location = str(row["location"])
        recording = str(row["recording"])
        scene_name, scene_meta = resolve_scene(location, recording, scene_index)
        if scene_meta is None:
            skipped.append(skipped_row(row, "source_scene_not_found", scene_name, None))
            continue

        first_frame = int_float(row.get("first_intersection_frame"))
        after_10m_frame = int_float(row.get("after_10m_frame"))
        if first_frame is None or after_10m_frame is None:
            skipped.append(skipped_row(row, "missing_key_frame", scene_name, scene_meta))
            continue
        if first_frame >= int(scene_meta["length_timesteps"]) or after_10m_frame >= int(
            scene_meta["length_timesteps"]
        ):
            skipped.append(skipped_row(row, "key_frame_outside_trajdata_scene", scene_name, scene_meta))
            continue

        start = max(0, first_frame - int(window_before_frames))
        end = min(
            int(scene_meta["length_timesteps"]) - 1,
            after_10m_frame + int(window_after_frames),
        )
        if end < start:
            end = start

        track_id = str(row["track_id"])
        scenario_id = (
            f"SIND_{location.upper()}_REDLIGHT_OFFSET_"
            f"{source_slug(recording)}_EGO{source_slug(track_id)}"
        )
        labels.append(
            {
                "scenario_id": scenario_id,
                "dataset": "sind",
                "location": location,
                "source_scene_name": scene_name,
                "scene_index": int(scene_meta["scene_index"]),
                "time_window": {
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "duration_sec": round((int(end) - int(start) + 1) * 0.1, 3),
                },
                "agents": {"ego_id": track_id},
                "semantic_tags": [
                    "red_light_running_stopline_abc",
                    "signal_time_offset_corrected",
                ],
                "semantics": clean_json(
                    {
                        "type": "red_light_running_stopline_abc",
                        "agent_type": row.get("agent_type"),
                        "movement": row.get("movement"),
                        "entrance_approach": row.get("entrance_approach"),
                        "exit_approach": row.get("exit_approach"),
                        "stop_line_key": row.get("stop_line_key"),
                        "signal_column": row.get("signal_column"),
                        "signal_time_offset_ms": float_or_none(row.get("signal_time_offset_ms")),
                        "first_intersection_frame": first_frame,
                        "last_intersection_frame": int_float(row.get("last_intersection_frame")),
                        "after_10m_frame": after_10m_frame,
                        "first_intersection_timestamp_ms": float_or_none(
                            row.get("first_intersection_timestamp_ms")
                        ),
                        "last_intersection_timestamp_ms": float_or_none(
                            row.get("last_intersection_timestamp_ms")
                        ),
                        "after_10m_timestamp_ms": float_or_none(
                            row.get("after_10m_timestamp_ms")
                        ),
                        "first_intersection_signal": row.get("first_intersection_signal"),
                        "last_intersection_signal": row.get("last_intersection_signal"),
                        "after_10m_signal": row.get("after_10m_signal"),
                        "red_light_violation": parse_bool(row.get("red_light_violation")),
                    }
                ),
                "source": clean_json(
                    {
                        "path": str(input_csv),
                        "recording": recording,
                        "track_id": track_id,
                        "source_result": "chengxiang_sind_six_intersections_results_no_frames.zip",
                    }
                ),
            }
        )

    labels = sorted(
        labels,
        key=lambda item: (
            item["location"],
            item["source_scene_name"],
            item["time_window"]["start_frame"],
            item["scenario_id"],
        ),
    )
    return labels, skipped


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_skipped_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def skipped_row(
    row: Dict[str, Any],
    reason: str,
    scene_name: str,
    scene_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "reason": reason,
        "location": row.get("location", ""),
        "recording": row.get("recording", ""),
        "track_id": row.get("track_id", ""),
        "agent_type": row.get("agent_type", ""),
        "movement": row.get("movement", ""),
        "resolved_scene_name": scene_name,
        "resolved_scene_index": "" if scene_meta is None else scene_meta.get("scene_index", ""),
        "resolved_scene_length": "" if scene_meta is None else scene_meta.get("length_timesteps", ""),
        "first_intersection_frame": row.get("first_intersection_frame", ""),
        "last_intersection_frame": row.get("last_intersection_frame", ""),
        "after_10m_frame": row.get("after_10m_frame", ""),
        "signal_time_offset_ms": row.get("signal_time_offset_ms", ""),
    }


def resolve_scene(
    location: str,
    recording: str,
    scene_index: Dict[str, Dict[str, Dict[str, Any]]],
) -> tuple[str, Optional[Dict[str, Any]]]:
    candidates = [f"{location}_{recording}"]
    if location == "cc":
        suffix = recording.rsplit("_", 1)[-1]
        if suffix == "xx":
            candidates.append("cc_xx cc")
        else:
            number = int_float(suffix)
            if number is not None:
                candidates.append(f"cc_{number} cc")
    for candidate in candidates:
        meta = scene_index.get(location, {}).get(candidate)
        if meta is not None:
            return candidate, meta
    return candidates[0], None


def int_float(value: Any) -> Optional[int]:
    number = float_or_none(value)
    if number is None:
        return None
    return int(round(number))


def float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    main()
