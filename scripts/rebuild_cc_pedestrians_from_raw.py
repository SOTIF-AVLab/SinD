#!/usr/bin/env python
"""Reinject Changchun pedestrian tracks into the legacy SinD cc pkls.

The current `datasets/SinD_dataset/cc/tp_info_cc.pkl` omits pedestrian tracks
even though the original CSV release contains `Ped_smoothed_tracks.csv` for all
Changchun recordings. This script keeps the existing vehicle content untouched
and only refreshes pedestrian entries in both `tp_info_cc.pkl` and
`frame_data_cc.pkl`.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd


RAW_TO_SCENE_ID: Mapping[str, str] = {
    "changchun_pudong_507_009": "9 cc",
    "changchun_pudong_507_010": "10 cc",
    "changchun_pudong_507_011": "11 cc",
    "changchun_pudong_523_005": "5 cc",
    "changchun_pudong_523_006": "6 cc",
    "changchun_pudong_523_007": "7 cc",
    "changchun_pudong_523_008": "8 cc",
    "changchun_pudong_xx": "xx cc",
}


def _is_pedestrian_track(tp_data: Dict[str, Any]) -> bool:
    cls = str(tp_data.get("Class", "")).strip().lower()
    typ = str(tp_data.get("Type", "")).strip().lower()
    if cls == "pedestrian" or typ == "pedestrian":
        return True
    state = tp_data.get("State")
    if isinstance(state, pd.DataFrame) and "agent_type" in state.columns:
        values = state["agent_type"].dropna().astype(str).str.lower().unique().tolist()
        return values == ["pedestrian"] or "pedestrian" in values
    return False


def _is_pedestrian_frame_entry(entry: Dict[str, Any]) -> bool:
    info = entry.get("vehicle_info", {})
    if not isinstance(info, dict):
        return False
    return str(info.get("agent_type", "")).strip().lower() == "pedestrian"


def _build_tp_data(track_id: str, state_df: pd.DataFrame) -> Dict[str, Any]:
    state_df = state_df.sort_values("frame_id").reset_index(drop=True).copy()
    first_ts = float(state_df["timestamp_ms"].iloc[0])
    last_ts = float(state_df["timestamp_ms"].iloc[-1])
    return {
        "State": state_df,
        "ID": track_id,
        "InitialFrame": first_ts,
        "FinalFrame": last_ts,
        "Frame_nums": int(len(state_df)),
        "Width": 0.7,
        "Length": 0.7,
        "Class": "pedestrian",
        "Type": "pedestrian",
        "cardinal direction": "",
        "x_y": "",
    }


def _append_frame_entries(
    frame_scene: Dict[int, List[Dict[str, Any]]], track_id: str, state_df: pd.DataFrame
) -> None:
    for row in state_df.itertuples(index=False):
        frame_id = int(row.frame_id)
        frame_scene.setdefault(frame_id, []).append(
            {
                "tp_id": track_id,
                "vehicle_info": {
                    "track_id": row.track_id,
                    "frame_id": row.frame_id,
                    "timestamp_ms": row.timestamp_ms,
                    "agent_type": row.agent_type,
                    "x": row.x,
                    "y": row.y,
                    "vx": row.vx,
                    "vy": row.vy,
                    "ax": row.ax,
                    "ay": row.ay,
                },
            }
        )


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, payload: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def rebuild_cc_pedestrians(dataset_dir: Path, raw_root: Path) -> None:
    cc_dir = dataset_dir / "cc"
    tp_path = cc_dir / "tp_info_cc.pkl"
    frame_path = cc_dir / "frame_data_cc.pkl"
    tp_info = _load_pickle(tp_path)
    frame_data = _load_pickle(frame_path)

    for scene_id in RAW_TO_SCENE_ID.values():
        scene_tp = tp_info.setdefault(scene_id, {})
        for key in [k for k, v in scene_tp.items() if _is_pedestrian_track(v)]:
            del scene_tp[key]

        scene_frames = frame_data.setdefault(scene_id, {})
        for frame_id in list(scene_frames.keys()):
            entries = scene_frames[frame_id]
            if not isinstance(entries, list):
                continue
            kept = [entry for entry in entries if not _is_pedestrian_frame_entry(entry)]
            if kept:
                scene_frames[frame_id] = kept
            else:
                del scene_frames[frame_id]

    injected_tracks = 0
    for raw_folder, scene_id in RAW_TO_SCENE_ID.items():
        csv_path = raw_root / raw_folder / "Ped_smoothed_tracks.csv"
        ped_df = pd.read_csv(csv_path)
        scene_tp = tp_info.setdefault(scene_id, {})
        scene_frames = frame_data.setdefault(scene_id, {})
        for track_id, track_df in ped_df.groupby("track_id", sort=False):
            track_key = str(track_id)
            state_df = track_df[
                ["track_id", "frame_id", "timestamp_ms", "agent_type", "x", "y", "vx", "vy", "ax", "ay"]
            ].copy()
            scene_tp[track_key] = _build_tp_data(track_key, state_df)
            _append_frame_entries(scene_frames, track_key, state_df)
            injected_tracks += 1

    _save_pickle(tp_path, tp_info)
    _save_pickle(frame_path, frame_data)
    print(f"Injected {injected_tracks} Changchun pedestrian tracks into {tp_path} and {frame_path}.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/SinD_dataset"),
        help="Processed SinD dataset root containing cc/tp_info_cc.pkl.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("Data/Changchun"),
        help="Raw Changchun_Pudong folder with Ped_smoothed_tracks.csv files.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rebuild_cc_pedestrians(
        dataset_dir=args.dataset_dir.expanduser().resolve(),
        raw_root=args.raw_root.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
