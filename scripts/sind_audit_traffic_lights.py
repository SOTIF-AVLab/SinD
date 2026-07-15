#!/usr/bin/env python3
"""Audit optional SinD traffic-light CSV integration coverage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

from trajdata.dataset_specific.sind.sind_traffic_lights import (
    LOCATION_TO_RAW_FOLDER,
    build_traffic_light_dataframe,
    configured_traffic_light_root,
    find_traffic_light_csv,
)
from trajdata.dataset_specific.sind.sind_utils import SIND_LOCATIONS, SindObject


def _iter_target_locations(raw_locations: Iterable[str]) -> List[str]:
    return [loc for loc in raw_locations if loc in LOCATION_TO_RAW_FOLDER]


def audit_with_sind_pkls(
    sind_data_dir: Path,
    traffic_light_root: Path | None,
    locations: Iterable[str],
) -> List[Dict]:
    sind_obj = SindObject(sind_data_dir, load_locations=list(locations))
    scene_dt = sind_obj.get_dt()
    reports: List[Dict] = []

    for location in locations:
        if location not in sind_obj.locations:
            reports.append(
                {
                    "location": location,
                    "status": "skipped",
                    "message": f"{location} not found in {sind_data_dir}",
                }
            )
            continue

        for scene_id in sind_obj._get_scene_names_from_pickle(location):
            scene_name = f"{location}_{scene_id}"
            scene_length = sind_obj.get_scene_length(scene_name)
            _, report = build_traffic_light_dataframe(
                scene_name=scene_name,
                location=location,
                scene_id=scene_id,
                scene_length=scene_length,
                scene_dt=scene_dt,
                root=traffic_light_root,
                pkl_root=sind_data_dir,
            )
            reports.append(report.to_dict())
        sind_obj.unload_city(location)

    return reports


def audit_raw_only(traffic_light_root: Path, locations: Iterable[str]) -> List[Dict]:
    reports: List[Dict] = []
    for location in locations:
        raw_folder = LOCATION_TO_RAW_FOLDER.get(location)
        if raw_folder is None:
            continue

        location_dir = traffic_light_root / raw_folder
        if not location_dir.exists():
            reports.append(
                {
                    "location": location,
                    "status": "missing_location_folder",
                    "message": str(location_dir),
                }
            )
            continue

        for scene_folder in sorted(p for p in location_dir.iterdir() if p.is_dir()):
            csv_path, _ = find_traffic_light_csv(
                traffic_light_root, location, scene_folder.name
            )
            reports.append(
                {
                    "location": location,
                    "scene_id": scene_folder.name,
                    "status": "found" if csv_path else "missing_csv",
                    "csv_path": str(csv_path) if csv_path else None,
                }
            )

    return reports


def summarize(reports: List[Dict]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for report in reports:
        location = report.get("location", "unknown")
        status = report.get("status", "unknown")
        summary.setdefault(location, {})
        summary[location][status] = summary[location].get(status, 0) + 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traffic-light-dir",
        default=None,
        help="Root containing raw SinD CSV folders; defaults to SIND_TRAFFIC_LIGHT_DIR.",
    )
    parser.add_argument(
        "--sind-data-dir",
        default=os.environ.get("SIND_DATA_DIR"),
        help="Optional pkl SinD root for scene-length-aware validation.",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=list(SIND_LOCATIONS),
        help="SinD locations to audit.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full audit report as JSON.",
    )
    args = parser.parse_args()

    traffic_light_root = configured_traffic_light_root(args.traffic_light_dir)
    if traffic_light_root is None and not args.sind_data_dir:
        raise SystemExit(
            "No traffic-light source found. Pass --sind-data-dir for local pkl "
            "or pass --traffic-light-dir / set SIND_TRAFFIC_LIGHT_DIR for raw CSV."
        )

    locations = _iter_target_locations(args.locations)
    if args.sind_data_dir:
        reports = audit_with_sind_pkls(Path(args.sind_data_dir), traffic_light_root, locations)
    else:
        reports = audit_raw_only(traffic_light_root, locations)

    result = {
        "traffic_light_root": str(traffic_light_root) if traffic_light_root is not None else None,
        "summary": summarize(reports),
        "reports": reports,
    }

    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Wrote audit report to {args.output_json}")


if __name__ == "__main__":
    main()
