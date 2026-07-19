#!/usr/bin/env python3
"""Regenerate SinD frame_data PKL from a processed tp_info PKL."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np


def calculate_heading(x: np.ndarray, y: np.ndarray, i: int, n_points: int = 5) -> float:
    if i + n_points >= len(x):
        n_points = len(x) - i - 1
    if n_points <= 0:
        return np.nan
    dx = 0.0
    dy = 0.0
    for j in range(i + 1, i + n_points + 1):
        dx += float(x[j] - x[i])
        dy += float(y[j] - y[i])
    return float(np.arctan2(dy / n_points, dx / n_points))


def fill_missing_heading(track: Dict[str, Any]) -> None:
    state = track.get("State")
    if state is None or state.empty or "heading_rad" not in state.columns:
        return
    if track.get("Type") in {"mv", "nmv"}:
        return

    x = state["x"].to_numpy()
    y = state["y"].to_numpy()
    previous = np.nan
    heading_values = state["heading_rad"].to_numpy(copy=True)
    for i, value in enumerate(heading_values):
        if not np.isnan(value):
            previous = value
            continue
        heading = calculate_heading(x, y, i)
        if np.isnan(heading):
            heading = previous
        else:
            previous = heading
        heading_values[i] = heading
    state.loc[:, "heading_rad"] = heading_values


def tp_info_to_frame_data(tp_info: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[int, list]]:
    frame_data: Dict[str, Dict[int, list]] = {}
    for scene_id, scene_tracks in tp_info.items():
        frame_data[scene_id] = {}
        for tp_id, track in scene_tracks.items():
            fill_missing_heading(track)
            state = track.get("State")
            if state is None or state.empty:
                continue
            if "frame_id" not in state.columns:
                raise ValueError(f"Track {tp_id} in scene {scene_id} has no frame_id column.")
            for row in state.sort_values("frame_id").to_dict(orient="records"):
                frame_id = int(row["frame_id"])
                frame_data[scene_id].setdefault(frame_id, []).append(
                    {"tp_id": tp_id, "vehicle_info": row}
                )
        frame_data[scene_id] = dict(sorted(frame_data[scene_id].items()))
    return frame_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.tp_info.open("rb") as f:
        tp_info = pickle.load(f)
    frame_data = tp_info_to_frame_data(tp_info)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(frame_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
