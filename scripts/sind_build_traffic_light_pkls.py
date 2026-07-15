#!/usr/bin/env python3
"""Build local SinD traffic-light pkl files from the original CSV release."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajdata.dataset_specific.sind.sind_traffic_lights import (  # noqa: E402
    DEFAULT_MAPPING_PATH,
    LOCATION_TO_RAW_FOLDER,
    TRAFFIC_LIGHT_PKL_VERSION,
    _build_traffic_light_dataframe_from_csv,
    configured_traffic_light_root,
    load_signal_time_offsets,
    signal_time_offset_for_scene,
    traffic_light_pkl_path,
)
from trajdata.dataset_specific.sind.sind_utils import SIND_LOCATIONS, SindObject  # noqa: E402


def iter_locations(raw_locations: Iterable[str]) -> List[str]:
    return [loc for loc in raw_locations if loc in LOCATION_TO_RAW_FOLDER]


def mapping_fingerprint(mapping_path: Path) -> Dict[str, Any]:
    if not mapping_path.exists():
        return {"path": str(mapping_path), "exists": False}
    return {
        "path": str(mapping_path),
        "exists": True,
        "mtime_ns": mapping_path.stat().st_mtime_ns,
        "size_bytes": mapping_path.stat().st_size,
    }


def build_location_payload(
    sind_obj: SindObject,
    sind_data_dir: Path,
    traffic_light_root: Path,
    location: str,
    mapping_path: Path,
    signal_time_offsets: Dict[Any, float],
) -> Dict[str, Any]:
    scene_dt = sind_obj.get_dt()
    payload: Dict[str, Any] = {
        "version": TRAFFIC_LIGHT_PKL_VERSION,
        "location": location,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "traffic_light_root": str(traffic_light_root),
        "mapping": mapping_fingerprint(mapping_path),
        "scene_dt": float(scene_dt),
        "scenes": {},
    }

    for scene_id in sind_obj._get_scene_names_from_pickle(location):
        scene_name = f"{location}_{scene_id}"
        scene_length = sind_obj.get_scene_length(scene_name)
        signal_time_offset_ms = signal_time_offset_for_scene(
            signal_time_offsets,
            location,
            scene_id,
        )
        tls_df, report, raw_changes = _build_traffic_light_dataframe_from_csv(
            scene_name=scene_name,
            location=location,
            scene_id=scene_id,
            scene_length=scene_length,
            scene_dt=scene_dt,
            root=traffic_light_root,
            mapping_path=mapping_path,
            include_raw_changes=True,
            signal_time_offset_ms=signal_time_offset_ms,
        )
        scene_payload = {
            "scene_name": scene_name,
            "scene_id": scene_id,
            "scene_length": int(scene_length),
            "signal_time_offset_ms": float(signal_time_offset_ms),
            "csv_path": report.csv_path,
            "report": report.to_dict(),
            "raw_changes": raw_changes,
            "traffic_light_status": tls_df,
        }
        payload["scenes"][scene_id] = scene_payload
    return payload


def summarize_payload(payload: Dict[str, Any]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for scene_payload in payload.get("scenes", {}).values():
        status = scene_payload.get("report", {}).get("status", "unknown")
        summary[status] = summary.get(status, 0) + 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sind-data-dir", type=Path, required=True)
    parser.add_argument("--traffic-light-dir", type=Path, default=None)
    parser.add_argument("--locations", nargs="+", default=list(SIND_LOCATIONS))
    parser.add_argument("--mapping-path", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument(
        "--signal-offset-csv",
        type=Path,
        default=None,
        help="Optional CSV with location, scene_id/recording, and signal_time_offset_ms.",
    )
    parser.add_argument("--output-summary", type=Path, default=None)
    args = parser.parse_args()

    traffic_light_root = configured_traffic_light_root(args.traffic_light_dir)
    if traffic_light_root is None:
        raise SystemExit("No traffic-light root found. Pass --traffic-light-dir or set SIND_TRAFFIC_LIGHT_DIR.")

    locations = iter_locations(args.locations)
    signal_time_offsets = load_signal_time_offsets(args.signal_offset_csv)
    sind_obj = SindObject(args.sind_data_dir, load_locations=locations)
    summary_rows: List[Dict[str, Any]] = []

    for location in locations:
        if location not in sind_obj.locations:
            summary_rows.append({"location": location, "status": "skipped", "message": "location not found"})
            continue
        print(f"Building traffic-light pkl for {location}...", flush=True)
        payload = build_location_payload(
            sind_obj=sind_obj,
            sind_data_dir=args.sind_data_dir,
            traffic_light_root=traffic_light_root,
            location=location,
            mapping_path=args.mapping_path,
            signal_time_offsets=signal_time_offsets,
        )
        output_path = traffic_light_pkl_path(args.sind_data_dir, location)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        status_counts = summarize_payload(payload)
        summary_row = {
            "location": location,
            "output_path": str(output_path),
            "num_scenes": len(payload.get("scenes", {})),
            "num_offset_scenes": sum(
                1
                for scene_payload in payload.get("scenes", {}).values()
                if float(scene_payload.get("signal_time_offset_ms", 0.0) or 0.0) != 0.0
            ),
            **{f"status_{key}": value for key, value in sorted(status_counts.items())},
        }
        summary_rows.append(summary_row)
        print(json.dumps(summary_row, ensure_ascii=False), flush=True)
        sind_obj.unload_city(location)

    result = {
        "sind_data_dir": str(args.sind_data_dir),
        "traffic_light_root": str(traffic_light_root),
        "mapping_path": str(args.mapping_path),
        "signal_offset_csv": str(args.signal_offset_csv) if args.signal_offset_csv else None,
        "locations": summary_rows,
    }
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote summary to {args.output_summary}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
