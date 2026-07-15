#!/usr/bin/env python3
"""Convert one SinD CSV record folder into the processed SinD 2.0 PKL layout."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_EXTENTS = {
    "car": (4.5, 2.0),
    "truck": (10.0, 2.5),
    "bus": (9.0, 3.0),
    "motorcycle": (2.0, 0.8),
    "bicycle": (1.8, 0.6),
    "tricycle": (3.5, 1.5),
    "pedestrian": (0.7, 0.7),
    "mv": (4.5, 2.0),
    "nmv": (1.8, 0.6),
}


def broad_type(agent_type: str) -> str:
    value = str(agent_type).strip().lower()
    if value in {"car", "truck", "bus"}:
        return "mv"
    if value in {"bicycle", "tricycle", "motorcycle"}:
        return "nmv"
    return value or "unknown"


def read_meta(record_dir: Path, prefix: str) -> Dict[str, Dict[str, Any]]:
    candidates = [
        record_dir / f"{prefix}_tracks_meta.csv",
        record_dir / f"{prefix}_track_meta.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        id_col = "trackId" if "trackId" in df.columns else "track_id"
        return {str(row[id_col]): row.to_dict() for _, row in df.iterrows()}
    return {}


def normalize_state(df: pd.DataFrame, default_type: str) -> pd.DataFrame:
    state = df.copy()
    if "agent_type" not in state.columns:
        state["agent_type"] = default_type
    for col in ["vx", "vy", "ax", "ay"]:
        if col not in state.columns:
            state[col] = 0.0
    if "heading_rad" not in state.columns:
        if {"vx", "vy"}.issubset(state.columns):
            state["heading_rad"] = np.arctan2(state["vy"], state["vx"])
        elif "yaw_rad" in state.columns:
            state["heading_rad"] = state["yaw_rad"]
        else:
            state["heading_rad"] = np.nan
    if "yaw_rad" not in state.columns:
        state["yaw_rad"] = state["heading_rad"]
    if "timestamp_ms" not in state.columns:
        state["timestamp_ms"] = state["frame_id"].astype(float) * 100.0
    return state.sort_values("frame_id").reset_index(drop=True)


def load_track_file(
    path: Path,
    *,
    default_type: str,
    meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if df.empty:
        return {}
    if "track_id" not in df.columns:
        raise ValueError(f"{path} is missing required column: track_id")
    if "frame_id" not in df.columns:
        raise ValueError(f"{path} is missing required column: frame_id")

    tracks: Dict[str, Dict[str, Any]] = {}
    for raw_id, group in df.groupby("track_id", sort=True):
        track_id = str(raw_id)
        state = normalize_state(group, default_type)
        first_row = state.iloc[0]
        meta_row = meta.get(track_id, {})

        class_name = str(
            meta_row.get("class")
            or meta_row.get("Class")
            or first_row.get("agent_type")
            or default_type
        ).strip().lower()
        type_name = broad_type(class_name)

        length = float(
            first_row.get(
                "length",
                meta_row.get("length", DEFAULT_EXTENTS.get(class_name, DEFAULT_EXTENTS[type_name])[0]),
            )
        )
        width = float(
            first_row.get(
                "width",
                meta_row.get("width", DEFAULT_EXTENTS.get(class_name, DEFAULT_EXTENTS[type_name])[1]),
            )
        )
        state["length"] = state.get("length", length)
        state["width"] = state.get("width", width)

        tracks[track_id] = {
            "Type": type_name,
            "Class": class_name,
            "Length": length,
            "Width": width,
            "InitialFrame": int(state["frame_id"].min()),
            "FinalFrame": int(state["frame_id"].max()),
            "State": state,
        }
    return tracks


def build_frame_data(tp_scene: Dict[str, Dict[str, Any]]) -> Dict[int, list]:
    frame_data: Dict[int, list] = {}
    for tp_id, track in tp_scene.items():
        state = track["State"]
        for row in state.to_dict(orient="records"):
            frame_id = int(row["frame_id"])
            frame_data.setdefault(frame_id, []).append(
                {"tp_id": tp_id, "vehicle_info": row}
            )
    return dict(sorted(frame_data.items(), key=lambda item: item[0]))


def iter_track_sources(record_dir: Path) -> Iterable[tuple[Path, str, Dict[str, Dict[str, Any]]]]:
    yield record_dir / "Veh_smoothed_tracks.csv", "car", read_meta(record_dir, "Veh")
    yield record_dir / "Ped_smoothed_tracks.csv", "pedestrian", read_meta(record_dir, "Ped")


def convert_record(record_dir: Path, scene_id: str) -> tuple[dict, dict]:
    tp_scene: Dict[str, Dict[str, Any]] = {}
    for path, default_type, meta in iter_track_sources(record_dir):
        for track_id, track in load_track_file(path, default_type=default_type, meta=meta).items():
            unique_id = track_id
            if unique_id in tp_scene:
                unique_id = f"{default_type}_{track_id}"
            tp_scene[unique_id] = track
    if not tp_scene:
        raise ValueError(f"No tracks found under {record_dir}")
    return {scene_id: tp_scene}, {scene_id: build_frame_data(tp_scene)}


def merge_or_replace(path: Path, scene_id: str, payload: dict, merge: bool) -> dict:
    if merge and path.exists():
        with path.open("rb") as f:
            existing = pickle.load(f)
        existing[scene_id] = payload[scene_id]
        return existing
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge the scene into existing location PKLs instead of replacing them.",
    )
    args = parser.parse_args()

    tp_info, frame_data = convert_record(args.record_dir, args.scene_id)
    location_dir = args.output_dir / args.location
    location_dir.mkdir(parents=True, exist_ok=True)

    tp_path = location_dir / f"tp_info_{args.location}.pkl"
    frame_path = location_dir / f"frame_data_{args.location}.pkl"
    tp_payload = merge_or_replace(tp_path, args.scene_id, tp_info, args.merge)
    frame_payload = merge_or_replace(frame_path, args.scene_id, frame_data, args.merge)

    with tp_path.open("wb") as f:
        pickle.dump(tp_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with frame_path.open("wb") as f:
        pickle.dump(frame_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {tp_path}")
    print(f"Wrote {frame_path}")


if __name__ == "__main__":
    main()
