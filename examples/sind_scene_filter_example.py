"""Example usage for SinD scene filtering helpers.

Run from the repository root after installing the project in editable mode:

    python examples/sind_scene_filter_example.py --data-dir datasets/SinD_dataset --city cc
"""

import argparse
import pickle
import sys
import types
from pathlib import Path

import pandas as pd

from trajdata.dataset_specific.sind.scene_filters import (
    build_area_boundary,
    build_lane_area,
    is_following_vehicle,
    is_position_invalid,
    is_vehicle_static,
    load_curbstone_points,
)


def _install_legacy_pandas_index_alias() -> None:
    """Allow old pickles that reference pandas.core.indexes.numeric.Int64Index."""
    numeric_module = types.ModuleType("pandas.core.indexes.numeric")
    numeric_module.Int64Index = pd.Index
    sys.modules["pandas.core.indexes.numeric"] = numeric_module


def load_city_data(data_dir: Path, city: str):
    city_dir = data_dir / city
    with (city_dir / f"frame_data_{city}.pkl").open("rb") as f:
        frame_data = pickle.load(f)

    with (city_dir / f"tp_info_{city}.pkl").open("rb") as f:
        tp_info = pickle.load(f)

    return frame_data, tp_info


def first_frame_with_offset(frame_data, scene_id, offset: int):
    frame_ids = list(frame_data[scene_id].keys())
    return frame_ids[min(offset, len(frame_ids) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SinD scene filter examples.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/SinD_dataset"),
        help="Root directory containing SinD pkl city folders.",
    )
    parser.add_argument("--city", default="cc", help="SinD city/location id.")
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=100,
        help="Frame offset sampled from the first scene for frame-level examples.",
    )
    args = parser.parse_args()

    _install_legacy_pandas_index_alias()

    frame_data, tp_info = load_city_data(args.data_dir, args.city)
    city_points = load_curbstone_points()
    outer_path, inner_paths = build_area_boundary(city_points, args.city)
    lane_area = build_lane_area(city_points, args.city)

    scene_id = list(frame_data.keys())[0]
    frame_id = first_frame_with_offset(frame_data, scene_id, args.frame_offset)
    vehicles = frame_data[scene_id][frame_id]
    scene_tp_info = tp_info[scene_id]
    motor_types = {"car", "bus", "truck"}
    motor_vehicles = [
        vehicle
        for vehicle in vehicles
        if vehicle["vehicle_info"].get("agent_type") in motor_types
    ]

    results = {"static": 0, "following": 0, "position": 0, "passed": 0}
    for vehicle in motor_vehicles:
        trajectory = scene_tp_info.get(vehicle["tp_id"], {}).get("State")
        if trajectory is None:
            continue

        if is_vehicle_static(trajectory):
            results["static"] += 1
            continue

        if is_following_vehicle(vehicle, motor_vehicles, lane_area):
            results["following"] += 1
            continue

        if is_position_invalid(vehicle, outer_path, lane_area, inner_paths):
            results["position"] += 1
            continue

        results["passed"] += 1

    print(f"City: {args.city}")
    print(f"Scene: {scene_id}")
    print(f"Frame: {frame_id}")
    print(f"Motor vehicles: {len(motor_vehicles)}")
    print(f"Static filtered: {results['static']}")
    print(f"Following filtered: {results['following']}")
    print(f"Position filtered: {results['position']}")
    print(f"Passed: {results['passed']}")


if __name__ == "__main__":
    main()
