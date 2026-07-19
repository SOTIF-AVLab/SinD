#!/usr/bin/env python3
"""Generate visual sanity checks for optional SinD traffic-light integration."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trajdata.dataset_specific.sind.sind_traffic_lights import (  # noqa: E402
    DEFAULT_MAPPING_PATH,
    LOCATION_TO_RAW_FOLDER,
    TRAFFIC_LIGHT_FILE_PATTERNS,
    _light_key_from_column,
    build_traffic_light_dataframe,
    configured_traffic_light_root,
)
from trajdata.dataset_specific.sind.sind_utils import SIND_LOCATIONS, SindObject  # noqa: E402
from trajdata.maps import TrafficLightStatus  # noqa: E402

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "tricycle", "mv"}
STATUS_COLOR_MAP = ListedColormap(["#9ca3af", "#22c55e", "#ef4444", "#facc15"])
STATUS_NAME = {
    int(TrafficLightStatus.UNKNOWN): "UNKNOWN",
    int(TrafficLightStatus.GREEN): "GREEN",
    int(TrafficLightStatus.RED): "RED",
    int(TrafficLightStatus.YELLOW): "YELLOW",
}


def sanitize_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    return "_".join(part for part in safe.split("_") if part)


def iter_locations(raw_locations: Iterable[str]) -> List[str]:
    return [loc for loc in raw_locations if loc in LOCATION_TO_RAW_FOLDER]


def load_mapping(mapping_path: Path = DEFAULT_MAPPING_PATH) -> Dict[str, Any]:
    if not mapping_path.exists():
        return {}
    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_light_keys(traffic_light_root: Path, location: str) -> List[str]:
    raw_folder = LOCATION_TO_RAW_FOLDER.get(location)
    if raw_folder is None:
        return []
    location_dir = traffic_light_root / raw_folder
    if not location_dir.exists():
        return []
    keys = set()
    for pattern in TRAFFIC_LIGHT_FILE_PATTERNS:
        for csv_path in location_dir.glob(f"*/{pattern}"):
            try:
                header = pd.read_csv(csv_path, nrows=0).columns
            except Exception:
                continue
            for column in header:
                if column in {"RawFrameID", "timestamp(ms)"}:
                    continue
                if "traffic" in column.lower() and "light" in column.lower():
                    kind, light_idx = _light_key_from_column(column)
                    keys.add(f"{kind}:{light_idx}")
    return sorted(keys, key=lambda key: (key.split(":")[0], int(key.split(":")[1])))


def mapping_coverage_rows(
    traffic_light_root: Optional[Path],
    locations: Sequence[str],
    mapping_path: Path = DEFAULT_MAPPING_PATH,
) -> List[Dict[str, Any]]:
    mapping = load_mapping(mapping_path)
    rows: List[Dict[str, Any]] = []
    for location in locations:
        light_keys = discover_light_keys(traffic_light_root, location) if traffic_light_root is not None else []
        loc_mapping = mapping.get("locations", {}).get(location, {}).get("light_to_lanes", {})
        if not light_keys:
            light_keys = sorted(loc_mapping.keys(), key=lambda key: (key.split(":")[0], int(key.split(":")[1])))
        mapped_lights = [key for key in light_keys if loc_mapping.get(key)]
        mapped_lanes = sorted({lane_id for key in mapped_lights for lane_id in loc_mapping.get(key, [])})
        rows.append(
            {
                "location": location,
                "total_lights": len(light_keys),
                "mapped_lights": len(mapped_lights),
                "mapped_lanes": len(mapped_lanes),
                "coverage": (len(mapped_lights) / len(light_keys)) if light_keys else 0.0,
            }
        )
    return rows


def load_vehicle_states(scenario: Mapping[str, Any]) -> pd.DataFrame:
    records: List[pd.DataFrame] = []
    for agent_id, track in scenario["tp_info"].items():
        class_name = str(track.get("Class", track.get("Type", ""))).lower()
        type_name = str(track.get("Type", "")).lower()
        if class_name not in VEHICLE_CLASSES and type_name not in VEHICLE_CLASSES:
            continue

        state = track.get("State")
        if state is None or state.empty:
            continue
        required = {"frame_id", "x", "y"}
        if not required.issubset(state.columns):
            continue

        state = state.copy()
        state["agent_id"] = str(agent_id)
        if {"vx", "vy"}.issubset(state.columns):
            state["speed"] = np.hypot(state["vx"].to_numpy(), state["vy"].to_numpy())
        else:
            state["speed"] = np.nan
        records.append(state[["agent_id", "frame_id", "x", "y", "speed"]])

    if not records:
        return pd.DataFrame(columns=["agent_id", "frame_id", "x", "y", "speed"])
    return pd.concat(records, ignore_index=True)


def infer_core_geometry(vehicle_df: pd.DataFrame) -> Dict[str, float]:
    if vehicle_df.empty:
        return {"center_x": 0.0, "center_y": 0.0, "core_radius": 20.0, "approach_radius": 50.0}

    xy = vehicle_df[["x", "y"]].to_numpy(dtype=float)
    center = np.nanmedian(xy, axis=0)
    dist = np.linalg.norm(xy - center, axis=1)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        core_radius = 20.0
    else:
        core_radius = float(np.clip(np.nanpercentile(dist, 35), 12.0, 35.0))
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "core_radius": core_radius,
        "approach_radius": core_radius + 32.0,
    }


def compute_flow_metrics(
    vehicle_df: pd.DataFrame,
    scene_length: int,
    scene_dt: float,
    bin_seconds: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    num_bins = max(1, int(math.ceil(scene_length * scene_dt / bin_seconds)))
    metrics = pd.DataFrame({"bin": np.arange(num_bins, dtype=int)})
    metrics["time_s"] = metrics["bin"] * bin_seconds
    for col in [
        "present_vehicles",
        "core_vehicles",
        "approach_stopped",
        "approach_moving",
        "core_entries",
        "mean_speed",
    ]:
        metrics[col] = 0.0

    geom = infer_core_geometry(vehicle_df)
    if vehicle_df.empty:
        return metrics, geom

    df = vehicle_df.copy()
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["frame_id", "x", "y"])
    df["frame_id"] = df["frame_id"].astype(int)
    df = df[(df["frame_id"] >= 0) & (df["frame_id"] < scene_length)]
    if df.empty:
        return metrics, geom

    dist = np.hypot(df["x"] - geom["center_x"], df["y"] - geom["center_y"])
    df["in_core"] = dist <= geom["core_radius"]
    df["in_approach"] = (dist > geom["core_radius"]) & (dist <= geom["approach_radius"])
    df["stopped"] = df["speed"].fillna(0.0) < 0.5
    df["moving"] = df["speed"].fillna(0.0) > 1.0
    df["bin"] = np.floor((df["frame_id"] * scene_dt) / bin_seconds).astype(int)
    df = df[(df["bin"] >= 0) & (df["bin"] < num_bins)]

    grouped = df.groupby("bin")
    present = grouped["agent_id"].nunique()
    core = df[df["in_core"]].groupby("bin")["agent_id"].nunique()
    stopped = df[df["in_approach"] & df["stopped"]].groupby("bin")["agent_id"].nunique()
    moving = df[df["in_approach"] & df["moving"]].groupby("bin")["agent_id"].nunique()
    mean_speed = grouped["speed"].mean()

    metrics["present_vehicles"] = metrics["bin"].map(present).fillna(0.0).to_numpy()
    metrics["core_vehicles"] = metrics["bin"].map(core).fillna(0.0).to_numpy()
    metrics["approach_stopped"] = metrics["bin"].map(stopped).fillna(0.0).to_numpy()
    metrics["approach_moving"] = metrics["bin"].map(moving).fillna(0.0).to_numpy()
    metrics["mean_speed"] = metrics["bin"].map(mean_speed).fillna(0.0).to_numpy()

    entry_bins: List[int] = []
    for _, track in df.sort_values(["agent_id", "frame_id"]).groupby("agent_id"):
        inside = track["in_core"].to_numpy(dtype=bool)
        if inside.size == 0:
            continue
        entered = inside & np.r_[True, ~inside[:-1]]
        entry_bins.extend(track.loc[entered, "bin"].astype(int).tolist())
    if entry_bins:
        counts = pd.Series(entry_bins).value_counts()
        metrics["core_entries"] = metrics["bin"].map(counts).fillna(0.0).to_numpy()

    return metrics, geom


def tls_phase_matrix(tls_df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    frame = tls_df.reset_index().drop_duplicates(["lane_id", "scene_ts"], keep="last")
    table = frame.pivot(index="lane_id", columns="scene_ts", values="status")
    table = table.sort_index(axis=0).sort_index(axis=1)
    arr = table.fillna(int(TrafficLightStatus.UNKNOWN)).to_numpy(dtype=int)
    known = {
        int(TrafficLightStatus.UNKNOWN),
        int(TrafficLightStatus.GREEN),
        int(TrafficLightStatus.RED),
        int(TrafficLightStatus.YELLOW),
    }
    arr = np.where(np.isin(arr, list(known)), arr, int(TrafficLightStatus.UNKNOWN))
    return [str(idx) for idx in table.index], arr


def per_second_signal_fractions(arr: np.ndarray, scene_dt: float, bin_seconds: float) -> pd.DataFrame:
    if arr.size == 0:
        return pd.DataFrame(columns=["bin", "time_s", "green_frac", "red_frac", "yellow_frac", "unknown_frac"])
    num_ts = arr.shape[1]
    bins = np.floor((np.arange(num_ts) * scene_dt) / bin_seconds).astype(int)
    rows = []
    for bin_id in range(int(bins.max()) + 1):
        sub = arr[:, bins == bin_id]
        if sub.size == 0:
            continue
        rows.append(
            {
                "bin": bin_id,
                "time_s": bin_id * bin_seconds,
                "green_frac": float(np.mean(sub == int(TrafficLightStatus.GREEN))),
                "red_frac": float(np.mean(sub == int(TrafficLightStatus.RED))),
                "yellow_frac": float(np.mean(sub == int(TrafficLightStatus.YELLOW))),
                "unknown_frac": float(np.mean(sub == int(TrafficLightStatus.UNKNOWN))),
            }
        )
    return pd.DataFrame(rows)


def phase_duration_summary(lane_ids: Sequence[str], arr: np.ndarray, scene_dt: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lane_id, values in zip(lane_ids, arr):
        total = max(1, values.size)
        row: Dict[str, Any] = {
            "lane_id": lane_id,
            "green_ratio": float(np.mean(values == int(TrafficLightStatus.GREEN))),
            "red_ratio": float(np.mean(values == int(TrafficLightStatus.RED))),
            "yellow_ratio": float(np.mean(values == int(TrafficLightStatus.YELLOW))),
            "unknown_ratio": float(np.mean(values == int(TrafficLightStatus.UNKNOWN))),
        }
        for status in [TrafficLightStatus.GREEN, TrafficLightStatus.RED, TrafficLightStatus.YELLOW]:
            durations = []
            mask = values == int(status)
            start = None
            for idx, flag in enumerate(np.r_[mask, False]):
                if flag and start is None:
                    start = idx
                elif not flag and start is not None:
                    durations.append((idx - start) * scene_dt)
                    start = None
            row[f"mean_{status.name.lower()}_s"] = float(np.mean(durations)) if durations else 0.0
            row[f"num_{status.name.lower()}_phases"] = len(durations)
        row["num_timesteps"] = total
        rows.append(row)
    return rows


def transition_response_scores(
    lane_ids: Sequence[str],
    arr: np.ndarray,
    metrics: pd.DataFrame,
    scene_dt: float,
    bin_seconds: float,
    window_s: float,
) -> List[Dict[str, Any]]:
    if metrics.empty or arr.size == 0:
        return []

    metric = metrics.set_index("bin")["core_entries"].astype(float)
    window_bins = max(1, int(round(window_s / bin_seconds)))
    rows: List[Dict[str, Any]] = []

    for lane_id, values in zip(lane_ids, arr):
        lane_row: Dict[str, Any] = {"lane_id": lane_id}
        for target in [TrafficLightStatus.GREEN, TrafficLightStatus.RED]:
            starts = np.flatnonzero((values == int(target)) & np.r_[True, values[:-1] != int(target)])
            deltas = []
            for start_ts in starts:
                center_bin = int(math.floor((start_ts * scene_dt) / bin_seconds))
                before_bins = range(center_bin - window_bins, center_bin)
                after_bins = range(center_bin + 1, center_bin + 1 + window_bins)
                before = metric.reindex(before_bins, fill_value=0.0).mean()
                after = metric.reindex(after_bins, fill_value=0.0).mean()
                deltas.append(float(after - before))
            lane_row[f"{target.name.lower()}_starts"] = int(len(starts))
            lane_row[f"{target.name.lower()}_entry_delta"] = float(np.mean(deltas)) if deltas else 0.0
        rows.append(lane_row)

    return rows


def plot_phase_matrix(
    lane_ids: Sequence[str],
    arr: np.ndarray,
    scene_dt: float,
    output_path: Path,
    title: str,
    max_cols: int = 2400,
) -> None:
    if arr.shape[1] > max_cols:
        cols = np.linspace(0, arr.shape[1] - 1, max_cols).astype(int)
        plot_arr = arr[:, cols]
    else:
        plot_arr = arr

    fig_h = max(3.5, min(12.0, 0.35 * len(lane_ids) + 1.8))
    fig, ax = plt.subplots(figsize=(14, fig_h), constrained_layout=True)
    extent = [0, arr.shape[1] * scene_dt / 60.0, len(lane_ids), 0]
    ax.imshow(plot_arr, aspect="auto", interpolation="nearest", cmap=STATUS_COLOR_MAP, extent=extent, vmin=0, vmax=3)
    ax.set_title(title)
    ax.set_xlabel("Scene time (min)")
    ax.set_ylabel("Traffic light id / lane_id")
    ax.set_yticks(np.arange(len(lane_ids)) + 0.5)
    ax.set_yticklabels([label[-42:] for label in lane_ids], fontsize=8)
    handles = [
        plt.Line2D([0], [0], color="#9ca3af", lw=8, label="UNKNOWN"),
        plt.Line2D([0], [0], color="#22c55e", lw=8, label="GREEN"),
        plt.Line2D([0], [0], color="#ef4444", lw=8, label="RED"),
        plt.Line2D([0], [0], color="#facc15", lw=8, label="YELLOW"),
    ]
    ax.legend(handles=handles, loc="upper right", ncols=4, frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_response(
    metrics: pd.DataFrame,
    signal_frac: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    ax0, ax1 = axes
    if not signal_frac.empty:
        ax0.plot(signal_frac["time_s"] / 60.0, signal_frac["green_frac"], color="#16a34a", label="GREEN fraction")
        ax0.plot(signal_frac["time_s"] / 60.0, signal_frac["red_frac"], color="#dc2626", label="RED fraction")
        ax0.plot(signal_frac["time_s"] / 60.0, signal_frac["yellow_frac"], color="#ca8a04", label="YELLOW fraction")
        ax0.plot(signal_frac["time_s"] / 60.0, signal_frac["unknown_frac"], color="#6b7280", label="UNKNOWN fraction")
    ax0.set_ylim(-0.02, 1.02)
    ax0.set_ylabel("Light-state fraction")
    ax0.set_title(title)
    ax0.legend(loc="upper right", ncols=4)
    ax0.grid(alpha=0.25)

    if not metrics.empty:
        x_min = metrics["time_s"] / 60.0
        ax1.plot(x_min, metrics["core_entries"], color="#2563eb", label="Core entries / bin")
        ax1.plot(x_min, metrics["approach_stopped"], color="#f97316", label="Approach stopped vehicles")
        ax1.plot(x_min, metrics["core_vehicles"], color="#0f766e", label="Core occupancy")
        ax1b = ax1.twinx()
        ax1b.plot(x_min, metrics["mean_speed"], color="#111827", alpha=0.45, label="Mean speed")
        ax1b.set_ylabel("Mean speed (m/s)")
        lines, labels = ax1.get_legend_handles_labels()
        lines_b, labels_b = ax1b.get_legend_handles_labels()
        ax1.legend(lines + lines_b, labels + labels_b, loc="upper right", ncols=4)
    ax1.set_xlabel("Scene time (min)")
    ax1.set_ylabel("Vehicle count")
    ax1.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_scene_html(
    output_path: Path,
    scene_title: str,
    summary: Mapping[str, Any],
    phase_img_rel: str,
    response_img_rel: str,
    duration_rows: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
) -> None:
    top_response = sorted(response_rows, key=lambda r: abs(float(r.get("green_entry_delta", 0.0))), reverse=True)[:12]
    duration_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['lane_id']))}</td>"
        f"<td>{row['green_ratio']:.3f}</td>"
        f"<td>{row['red_ratio']:.3f}</td>"
        f"<td>{row['yellow_ratio']:.3f}</td>"
        f"<td>{row['unknown_ratio']:.3f}</td>"
        f"<td>{row['mean_green_s']:.1f}</td>"
        f"<td>{row['mean_red_s']:.1f}</td>"
        f"<td>{row['mean_yellow_s']:.1f}</td>"
        "</tr>"
        for row in duration_rows
    )
    response_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['lane_id']))}</td>"
        f"<td>{row.get('green_starts', 0)}</td>"
        f"<td>{float(row.get('green_entry_delta', 0.0)):.3f}</td>"
        f"<td>{row.get('red_starts', 0)}</td>"
        f"<td>{float(row.get('red_entry_delta', 0.0)):.3f}</td>"
        "</tr>"
        for row in top_response
    )
    summary_html = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in summary.items()
    )
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(scene_title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #111827; }}
    h1, h2 {{ margin-bottom: 0.3rem; }}
    table {{ border-collapse: collapse; margin: 14px 0 28px; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 8px; }}
    .note {{ color: #4b5563; max-width: 980px; }}
  </style>
</head>
<body>
  <h1>{html.escape(scene_title)}</h1>
  <p class="note">Synthetic IDs mean the raw SinD light number is cached, but not yet manually bound to a true map lane. Use phase timing and traffic-flow response here to validate temporal alignment first.</p>
  <h2>Scene Summary</h2>
  <table>{summary_html}</table>
  <h2>Phase Timeline</h2>
  <img src="{html.escape(phase_img_rel)}" alt="phase matrix">
  <h2>Traffic Flow Response</h2>
  <img src="{html.escape(response_img_rel)}" alt="traffic flow response">
  <h2>Per-Light Phase Durations</h2>
  <table>
    <tr><th>lane_id</th><th>green_ratio</th><th>red_ratio</th><th>yellow_ratio</th><th>unknown_ratio</th><th>mean_green_s</th><th>mean_red_s</th><th>mean_yellow_s</th></tr>
    {duration_html}
  </table>
  <h2>Coarse Response Scores</h2>
  <p class="note">Scores are global sanity checks: mean change in intersection-core entries after a light transition versus before it. They are not lane-specific until light-to-lane mapping is completed.</p>
  <table>
    <tr><th>lane_id</th><th>green_starts</th><th>green_entry_delta</th><th>red_starts</th><th>red_entry_delta</th></tr>
    {response_html}
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_index(
    output_dir: Path,
    summaries: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
) -> None:
    coverage_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['location']))}</td>"
        f"<td>{row['mapped_lights']}/{row['total_lights']}</td>"
        f"<td>{float(row['coverage']):.3f}</td>"
        f"<td>{row['mapped_lanes']}</td>"
        "</tr>"
        for row in coverage_rows
    )
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['location']))}</td>"
        f"<td><a href=\"{html.escape(str(row['html_path']))}\">{html.escape(str(row['scene_id']))}</a></td>"
        f"<td>{html.escape(str(row['status']))}</td>"
        f"<td>{row.get('num_lights', 0)}</td>"
        f"<td>{row.get('mapped_lane_ids', 0)}</td>"
        f"<td>{row.get('synthetic_lane_ids', 0)}</td>"
        f"<td>{float(row.get('unknown_ratio', 0.0)):.3f}</td>"
        f"<td>{html.escape(str(row.get('alignment', '')))}</td>"
        f"<td>{html.escape(str(row.get('message', '')))}</td>"
        "</tr>"
        for row in summaries
    )
    output_dir.joinpath("index.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SinD Traffic Light Validation</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>SinD Traffic Light Validation</h1>
  <p>Open a scene to inspect phase timelines and coarse traffic-flow response.</p>
  <h2>Mapping Coverage</h2>
  <table>
    <tr><th>location</th><th>mapped_lights</th><th>coverage</th><th>mapped_lanes</th></tr>
    {coverage_html}
  </table>
  <h2>Scene Validation</h2>
  <table>
    <tr><th>location</th><th>scene</th><th>status</th><th>lights</th><th>mapped_lane_ids</th><th>synthetic_lane_ids</th><th>unknown_ratio</th><th>alignment</th><th>message</th></tr>
    {rows}
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_csv(output_dir: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    if not summaries:
        return
    keys = sorted({key for row in summaries for key in row.keys() if key != "html_path"})
    with output_dir.joinpath("summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in keys})


def visualize_scene(
    sind_obj: SindObject,
    traffic_light_root: Optional[Path],
    location: str,
    scene_id: str,
    output_dir: Path,
    bin_seconds: float,
    response_window_s: float,
) -> Dict[str, Any]:
    scene_name = f"{location}_{scene_id}"
    scene_length = sind_obj.get_scene_length(scene_name)
    scenario = sind_obj.load_scenario(scene_name)
    tls_df, report = build_traffic_light_dataframe(
        scene_name=scene_name,
        location=location,
        scene_id=scene_id,
        scene_length=scene_length,
        scene_dt=sind_obj.get_dt(),
        root=traffic_light_root,
        pkl_root=sind_obj.dataset_path,
    )

    safe_scene = sanitize_filename(scene_name)
    summary: Dict[str, Any] = report.to_dict()
    summary["html_path"] = ""
    if tls_df is None:
        return summary

    vehicle_df = load_vehicle_states(scenario)
    metrics, geom = compute_flow_metrics(vehicle_df, scene_length, sind_obj.get_dt(), bin_seconds)
    lane_ids, arr = tls_phase_matrix(tls_df)
    signal_frac = per_second_signal_fractions(arr, sind_obj.get_dt(), bin_seconds)
    duration_rows = phase_duration_summary(lane_ids, arr, sind_obj.get_dt())
    response_rows = transition_response_scores(
        lane_ids, arr, metrics, sind_obj.get_dt(), bin_seconds, response_window_s
    )

    unknown_ratio = float(np.mean(arr == int(TrafficLightStatus.UNKNOWN))) if arr.size else 1.0
    mapped_lane_ids = [lane_id for lane_id in lane_ids if not lane_id.startswith("sind_tl:")]
    synthetic_lane_ids = [lane_id for lane_id in lane_ids if lane_id.startswith("sind_tl:")]
    summary.update(
        {
            "num_vehicles": int(vehicle_df["agent_id"].nunique()) if not vehicle_df.empty else 0,
            "unknown_ratio": unknown_ratio,
            "mapped_lane_ids": len(mapped_lane_ids),
            "synthetic_lane_ids": len(synthetic_lane_ids),
            "center_x": f"{geom['center_x']:.2f}",
            "center_y": f"{geom['center_y']:.2f}",
            "core_radius_m": f"{geom['core_radius']:.1f}",
        }
    )

    fig_dir = output_dir / "figures"
    scene_dir = output_dir / "scenes"
    phase_img = fig_dir / f"{safe_scene}_phase_matrix.png"
    response_img = fig_dir / f"{safe_scene}_response.png"
    scene_html = scene_dir / f"{safe_scene}.html"
    scene_dir.mkdir(parents=True, exist_ok=True)

    plot_phase_matrix(lane_ids, arr, sind_obj.get_dt(), phase_img, f"{scene_name}: Traffic-light phase matrix")
    plot_response(metrics, signal_frac, response_img, f"{scene_name}: Signal fractions vs. traffic response")

    write_scene_html(
        scene_html,
        scene_name,
        summary,
        os.path.relpath(phase_img, scene_html.parent),
        os.path.relpath(response_img, scene_html.parent),
        duration_rows,
        response_rows,
    )
    summary["html_path"] = os.path.relpath(scene_html, output_dir)
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
        default=os.environ.get("SIND_DATA_DIR", "datasets/SinD_dataset"),
        help="Root containing SinD pkl files.",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=list(SIND_LOCATIONS),
        help="Locations to visualize.",
    )
    parser.add_argument(
        "--scenes-per-location",
        type=int,
        default=0,
        help="Limit scenes per location for quick checks. 0 means all scenes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("risk_mining/output_sind_traffic_lights"),
        help="Output directory for HTML, CSV, JSON, and figures.",
    )
    parser.add_argument("--bin-seconds", type=float, default=1.0)
    parser.add_argument("--response-window-s", type=float, default=8.0)
    args = parser.parse_args()

    traffic_light_root = configured_traffic_light_root(args.traffic_light_dir)
    sind_data_dir = Path(args.sind_data_dir)
    if traffic_light_root is None and not sind_data_dir.exists():
        raise SystemExit("No traffic-light source found. Pass a valid --sind-data-dir or --traffic-light-dir.")

    locations = iter_locations(args.locations)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sind_obj = SindObject(sind_data_dir, load_locations=locations)
    summaries: List[Dict[str, Any]] = []
    for location in locations:
        if location not in sind_obj.locations:
            summaries.append(
                {
                    "location": location,
                    "scene_id": "",
                    "status": "skipped",
                    "message": f"{location} not found in {args.sind_data_dir}",
                    "html_path": "",
                }
            )
            continue

        scene_ids = sind_obj._get_scene_names_from_pickle(location)
        if args.scenes_per_location > 0:
            scene_ids = scene_ids[: args.scenes_per_location]

        for scene_id in scene_ids:
            print(f"Visualizing {location}_{scene_id}...", flush=True)
            summaries.append(
                visualize_scene(
                    sind_obj,
                    traffic_light_root,
                    location,
                    scene_id,
                    output_dir,
                    args.bin_seconds,
                    args.response_window_s,
                )
            )
        sind_obj.unload_city(location)

    coverage_rows = mapping_coverage_rows(traffic_light_root, locations)
    write_index(output_dir, summaries, coverage_rows)
    write_csv(output_dir, summaries)
    output_dir.joinpath("mapping_coverage.json").write_text(
        json.dumps(coverage_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_dir.joinpath("summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote traffic-light validation report to {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
