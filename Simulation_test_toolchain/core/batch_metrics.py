from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon

from Simulation_test_toolchain.core.records import AgentFrame, SimulationResult
from Simulation_test_toolchain.pickle_compat import load_legacy_pandas_pickle
LANE_MATCH_THRESHOLD_M = 2.0
LANE_SAMPLE_SPACING_M = 1.0
MIN_LANE_SPEED_MPS = 1.0
MIN_WRONG_WAY_DURATION_S = 1.0
WRONG_WAY_HEADING_THRESHOLD_DEG = 120.0
RISK_TTC_HORIZON_STEPS = 25
RISK_TTC_MAX_SECONDS = 2.5


class MetricContext:
    def __init__(self, data_dir: Path, dt: float) -> None:
        self.data_dir = data_dir
        self.dt = dt
        self._road_boundary_by_location: Dict[str, Any] = {}
        self._lane_refs_by_location: Dict[str, Any] = {}
        self._lane_index_by_location: Dict[str, Any] = {}
        self._lane_rule_by_location: Dict[str, Any] = {}
        self._roi_by_location: Dict[str, Polygon] = {}
        self._signal_tables: Dict[Tuple[str, str, int], Tuple[Optional[pd.DataFrame], str]] = {}

    def road_boundary(self, location: str) -> Optional[Any]:
        if location not in self._road_boundary_by_location:
            self._road_boundary_by_location[location] = _load_curbstone_boundary(
                location
            )
        return self._road_boundary_by_location[location]

    def lane_context(self, location: str) -> Tuple[Optional[Any], Optional[Any], Optional[Polygon]]:
        if location not in self._lane_index_by_location:
            try:
                from risk_mining.intersection_spatiotemporal_density import (
                    infer_intersection_core_roi,
                )
                from risk_mining.lateral_deviation_variance import load_lane_references
                from risk_mining.structured_violations_noncompliance import (
                    build_lane_indices,
                    build_lane_rule_indices,
                )

                references = load_lane_references(self.data_dir, [location])
                lane_refs = references[location]
                self._lane_refs_by_location[location] = lane_refs
                self._lane_index_by_location[location] = build_lane_indices(
                    references, spacing=LANE_SAMPLE_SPACING_M
                ).get(location)
                self._lane_rule_by_location[location] = build_lane_rule_indices(
                    references
                ).get(location)
                roi, _ = infer_intersection_core_roi(lane_refs)
                self._roi_by_location[location] = roi
            except Exception:
                self._lane_refs_by_location[location] = None
                self._lane_index_by_location[location] = None
                self._lane_rule_by_location[location] = None
                self._roi_by_location[location] = None
        return (
            self._lane_index_by_location[location],
            self._lane_rule_by_location[location],
            self._roi_by_location[location],
        )

    def signal_table(
        self,
        scene_name: str,
        location: str,
        scene_id: str,
        scene_length: int,
    ) -> Tuple[Optional[pd.DataFrame], str]:
        key = (location, scene_id, scene_length)
        if key not in self._signal_tables:
            try:
                from trajdata.dataset_specific.sind.sind_traffic_lights import (
                    build_traffic_light_dataframe,
                )

                table, report = build_traffic_light_dataframe(
                    scene_name=scene_name,
                    location=location,
                    scene_id=scene_id,
                    scene_length=scene_length,
                    scene_dt=self.dt,
                    pkl_root=self.data_dir,
                )
                self._signal_tables[key] = (
                    table,
                    report.status if report is not None else "unavailable",
                )
            except Exception as exc:
                self._signal_tables[key] = (None, f"error: {exc}")
        return self._signal_tables[key]


def compute_batch_metrics(
    result: SimulationResult,
    data_dir: Path,
    context: MetricContext,
) -> Dict[str, Any]:
    metadata = dict(result.metadata)
    location = str(metadata["location"])
    scene_name = str(metadata["scene_name"])
    scene_id = _scene_id_from_name(location, scene_name)
    ego_agent = str(metadata.get("active_ego_agent") or metadata["ego_agent"])
    scene_length = _raw_scene_length(data_dir, location, scene_id)

    frames_df = _frames_to_df(result.frames)
    ego_df = frames_df[frames_df["is_ego"]].sort_values("timestep")
    gt_df = _load_gt_track(data_dir, location, scene_id, ego_agent)

    metrics: Dict[str, Any] = {
        "location": location,
        "scene_name": scene_name,
        "scene_id": scene_id,
        "ego_agent": ego_agent,
        "num_logged_ego_frames": int(len(ego_df)),
        "num_steps": int(metadata.get("num_steps", max(len(ego_df) - 1, 0))),
    }
    metrics.update(_compute_ade_fde(ego_df, gt_df))
    metrics.update(_compute_collision_metrics(frames_df))
    metrics.update(_compute_risk_metrics(frames_df, context.dt))
    metrics.update(_compute_offroad_metrics(ego_df, context.road_boundary(location)))
    metrics.update(
        _compute_violation_metrics(
            ego_df=ego_df,
            context=context,
            location=location,
            scene_name=scene_name,
            scene_id=scene_id,
            scene_length=scene_length,
        )
    )
    metrics["violation"] = bool(
        metrics["wrong_way_violation"]
        or metrics["red_light_violation"]
        or metrics["lane_direction_rule_violation"]
    )
    return metrics


def save_metrics(path: Path, metadata: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    payload = {"metadata": dict(metadata), "metrics": dict(metrics)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_result_json(path: Path) -> SimulationResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = [AgentFrame(**frame) for frame in payload.get("frames", [])]
    return SimulationResult(metadata=payload.get("metadata", {}), frames=frames)


def _scene_id_from_name(location: str, scene_name: str) -> str:
    prefix = f"{location}_"
    return scene_name[len(prefix) :] if scene_name.startswith(prefix) else scene_name


def _frames_to_df(frames: Sequence[AgentFrame]) -> pd.DataFrame:
    rows = [asdict(frame) if isinstance(frame, AgentFrame) else dict(frame) for frame in frames]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows)


def _load_location_tp_info(data_dir: Path, location: str) -> Mapping[str, Any]:
    path = data_dir / location / f"tp_info_{location}.pkl"
    with path.open("rb") as handle:
        return load_legacy_pandas_pickle(handle)


def _raw_scene_length(data_dir: Path, location: str, scene_id: str) -> int:
    scene_tracks = _load_location_tp_info(data_dir, location)[scene_id]
    max_frame = 0
    for tp_data in scene_tracks.values():
        state = tp_data.get("State")
        if state is not None and not state.empty and "frame_id" in state.columns:
            max_frame = max(max_frame, int(state["frame_id"].max()))
    return max_frame + 1


def _load_gt_track(data_dir: Path, location: str, scene_id: str, agent_id: str) -> pd.DataFrame:
    scene_tracks = _load_location_tp_info(data_dir, location)[scene_id]
    tp_data = next(
        (value for key, value in scene_tracks.items() if str(key) == str(agent_id)),
        None,
    )
    if tp_data is None:
        return pd.DataFrame()
    state = tp_data.get("State")
    if state is None or state.empty:
        return pd.DataFrame()
    return state.copy().sort_values("frame_id")


def _compute_ade_fde(ego_df: pd.DataFrame, gt_df: pd.DataFrame) -> Dict[str, Any]:
    if ego_df.empty or gt_df.empty:
        return {
            "ADE": np.nan,
            "FDE": np.nan,
            "gt_aligned_frame_count": 0,
        }

    gt_xy = gt_df.set_index("frame_id")[["x", "y"]]
    rows: List[Tuple[float, float, float, float]] = []
    for row in ego_df.itertuples(index=False):
        timestep = int(row.timestep)
        if timestep not in gt_xy.index:
            continue
        gt_row = gt_xy.loc[timestep]
        if isinstance(gt_row, pd.DataFrame):
            gt_row = gt_row.iloc[0]
        rows.append((float(row.x), float(row.y), float(gt_row["x"]), float(gt_row["y"])))

    if not rows:
        return {
            "ADE": np.nan,
            "FDE": np.nan,
            "gt_aligned_frame_count": 0,
        }
    arr = np.asarray(rows, dtype=float)
    distances = np.linalg.norm(arr[:, :2] - arr[:, 2:], axis=1)
    return {
        "ADE": float(np.mean(distances)),
        "FDE": float(distances[-1]),
        "gt_aligned_frame_count": int(len(distances)),
    }


def _compute_collision_metrics(frames_df: pd.DataFrame) -> Dict[str, Any]:
    if frames_df.empty:
        return {
            "collision": False,
            "first_collision_timestep": None,
            "collision_agent_id": None,
        }

    for timestep, step_df in frames_df.groupby("timestep", sort=True):
        ego_rows = step_df[step_df["is_ego"]]
        if ego_rows.empty:
            continue
        ego_poly = _agent_box_polygon(ego_rows.iloc[0])
        for other in step_df[~step_df["is_ego"]].itertuples(index=False):
            other_poly = _agent_box_polygon(other)
            if ego_poly.intersects(other_poly):
                return {
                    "collision": True,
                    "first_collision_timestep": int(timestep),
                    "collision_agent_id": str(other.agent_name),
                }

    return {
        "collision": False,
        "first_collision_timestep": None,
        "collision_agent_id": None,
    }


def _compute_risk_metrics(frames_df: pd.DataFrame, dt: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "MinTTC": np.nan,
        "AveTTC": np.nan,
        "MRD": np.nan,
        "ARD": np.nan,
        "risk_frame_count": 0,
    }
    if frames_df.empty:
        return result

    frame_min_ttc: List[float] = []
    frame_min_dist: List[float] = []
    for _, step_df in frames_df.groupby("timestep", sort=True):
        ego_rows = step_df[step_df["is_ego"]]
        others = step_df[~step_df["is_ego"]]
        if ego_rows.empty or others.empty:
            continue
        ego = ego_rows.iloc[0]
        ego_poly = _agent_box_polygon(ego)
        min_ttc = RISK_TTC_MAX_SECONDS
        min_dist = float("inf")
        for other in others.itertuples(index=False):
            other_poly = _agent_box_polygon(other)
            min_dist = min(min_dist, float(ego_poly.distance(other_poly)))
            pair_ttc = _pair_rollout_ttc(ego, other, dt)
            min_ttc = min(min_ttc, pair_ttc)
        if np.isfinite(min_dist):
            frame_min_dist.append(min_dist)
            frame_min_ttc.append(min_ttc)

    if frame_min_dist:
        result["MRD"] = float(np.min(frame_min_dist))
        result["ARD"] = float(np.mean(frame_min_dist))
        result["MinTTC"] = float(np.min(frame_min_ttc))
        result["AveTTC"] = float(np.mean(frame_min_ttc))
        result["risk_frame_count"] = int(len(frame_min_dist))
    return result


def _pair_rollout_ttc(ego: Any, other: Any, dt: float) -> float:
    ego_state = _row_motion_state(ego)
    other_state = _row_motion_state(other)
    for step in range(RISK_TTC_HORIZON_STEPS + 1):
        ttc = step * dt
        ego_future = _rollout_constant_heading(ego_state, ttc)
        other_future = _rollout_constant_heading(other_state, ttc)
        if _agent_state_polygon(ego_future).intersects(
            _agent_state_polygon(other_future)
        ):
            return float(ttc)
    return RISK_TTC_MAX_SECONDS


def _row_motion_state(row: Any) -> Dict[str, float]:
    speed = float(_row_value(row, "speed"))
    heading = float(_row_value(row, "heading"))
    return {
        "x": float(_row_value(row, "x")),
        "y": float(_row_value(row, "y")),
        "heading": heading,
        "speed": speed if np.isfinite(speed) else 0.0,
        "length": max(float(_row_value(row, "length")), 0.1),
        "width": max(float(_row_value(row, "width")), 0.1),
    }


def _rollout_constant_heading(state: Mapping[str, float], ttc: float) -> Dict[str, float]:
    speed = float(state["speed"])
    heading = float(state["heading"])
    return {
        **state,
        "x": float(state["x"] + speed * math.cos(heading) * ttc),
        "y": float(state["y"] + speed * math.sin(heading) * ttc),
    }


def _agent_state_polygon(state: Mapping[str, float]) -> Polygon:
    return Polygon(
        _box_corners(
            float(state["x"]),
            float(state["y"]),
            float(state["heading"]),
            float(state["length"]),
            float(state["width"]),
        )
    )


def _compute_offroad_metrics(ego_df: pd.DataFrame, road_boundary: Optional[Any]) -> Dict[str, Any]:
    result = {
        "offroad_observable": road_boundary is not None,
        "offroad": False,
        "first_offroad_timestep": None,
        "offroad_frame_count": 0,
    }
    if ego_df.empty or road_boundary is None:
        return result

    for row in ego_df.itertuples(index=False):
        points = [(float(row.x), float(row.y))]
        points.extend(_agent_box_corners(row))
        is_offroad = any(not road_boundary.covers(Point(x, y)) for x, y in points)
        if is_offroad:
            result["offroad_frame_count"] += 1
            if result["first_offroad_timestep"] is None:
                result["first_offroad_timestep"] = int(row.timestep)

    result["offroad"] = bool(result["offroad_frame_count"] > 0)
    return result


def _compute_violation_metrics(
    ego_df: pd.DataFrame,
    context: MetricContext,
    location: str,
    scene_name: str,
    scene_id: str,
    scene_length: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "wrong_way_violation": False,
        "first_wrong_way_timestep": None,
        "lane_direction_rule_violation": False,
        "red_light_violation": False,
        "signal_observable": False,
        "red_light_status_at_entry": "unavailable",
        "red_light_entry_timestep": None,
        "violation_observable": False,
    }
    if len(ego_df) < 2:
        return result

    try:
        from risk_mining.structured_violations_noncompliance import (
            _actual_maneuver_from_roi_geometry,
            _allowed_movements_for_lane,
            _first_true_index,
            _last_true_index,
            _points_inside_polygon,
            _run_lengths,
            _stable_entry_lane,
            _status_name,
            query_lane_index,
        )
    except ImportError:
        return result
    from trajdata.maps import TrafficLightStatus

    lane_index, lane_rule_index, roi = context.lane_context(location)
    if lane_index is None or roi is None:
        return result
    result["violation_observable"] = True

    xy = ego_df[["x", "y"]].to_numpy(dtype=float)
    timesteps = ego_df["timestep"].to_numpy(dtype=int)
    vel = np.gradient(xy, context.dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    speed_safe = np.maximum(speed, 1e-6)
    vel_unit = np.divide(vel, speed_safe[:, None])
    distances, _, lane_ids, lane_dirs = query_lane_index(lane_index, xy)
    valid_lane = (distances <= LANE_MATCH_THRESHOLD_M) & np.isfinite(speed)
    moving = speed >= MIN_LANE_SPEED_MPS
    cos_lane = np.sum(vel_unit * lane_dirs, axis=1)
    roi_mask = _points_inside_polygon(xy, roi)

    cos_threshold = math.cos(math.radians(WRONG_WAY_HEADING_THRESHOLD_DEG))
    wrong_mask = valid_lane & (~roi_mask) & moving & (cos_lane < cos_threshold)
    min_wrong_pts = max(1, int(math.ceil(MIN_WRONG_WAY_DURATION_S / context.dt)))
    wrong_runs = [
        (start, end) for start, end in _run_lengths(wrong_mask) if end - start >= min_wrong_pts
    ]
    if wrong_runs:
        first_start, _ = wrong_runs[0]
        result["wrong_way_violation"] = True
        result["first_wrong_way_timestep"] = int(timesteps[first_start])

    entry_idx = _first_true_index(roi_mask & valid_lane)
    exit_idx = _last_true_index(roi_mask & valid_lane)
    if entry_idx is not None:
        min_run_pts = max(1, int(math.ceil(0.5 / context.dt)))
        entry_lane_id, _ = _stable_entry_lane(
            lane_ids,
            valid_lane,
            moving,
            entry_idx,
            min_run_pts=min_run_pts,
            lookback_pts=max(min_run_pts * 2, int(round(8.0 / context.dt))),
        )
        actual_maneuver = _actual_maneuver_from_roi_geometry(
            xy,
            roi_mask,
            entry_idx,
            exit_idx,
            outside_pts=max(5, int(round(2.5 / context.dt))),
        )
        allowed = _allowed_movements_for_lane(entry_lane_id, lane_rule_index)
        result["entry_lane_id"] = entry_lane_id
        result["actual_maneuver"] = actual_maneuver
        result["allowed_movements"] = ";".join(allowed)
        if allowed and actual_maneuver != "unknown" and actual_maneuver not in allowed:
            result["lane_direction_rule_violation"] = True

        tls_df, signal_status = context.signal_table(
            scene_name=scene_name,
            location=location,
            scene_id=scene_id,
            scene_length=scene_length,
        )
        result["signal_table_status"] = signal_status
        status, signal_lane_id = _status_for_entry(
            tls_df,
            [entry_lane_id, str(lane_ids[entry_idx])],
            int(timesteps[entry_idx]),
        )
        if status is not None:
            result["signal_observable"] = True
            result["red_light_status_at_entry"] = _status_name(status)
            result["signal_lane_id"] = signal_lane_id
            if status == int(TrafficLightStatus.RED):
                result["red_light_violation"] = True
                result["red_light_entry_timestep"] = int(timesteps[entry_idx])

    return result


def _status_for_entry(
    tls_df: Optional[pd.DataFrame],
    lane_ids: Sequence[str],
    scene_ts: int,
) -> Tuple[Optional[int], str]:
    if tls_df is None:
        return None, ""
    try:
        from risk_mining.structured_violations_noncompliance import _status_at_scene_ts
    except ImportError:
        return None, ""

    for lane_id in lane_ids:
        if not lane_id:
            continue
        status = _status_at_scene_ts(tls_df, str(lane_id), scene_ts)
        if status is not None:
            return status, str(lane_id)
    return None, ""


def _load_curbstone_boundary(location: str) -> Optional[Any]:
    try:
        from trajdata.dataset_specific.sind.scene_filters import load_curbstone_points

        city_points = load_curbstone_points()
        curbstone = city_points[location]
        points = []
        for xs, ys in zip(curbstone["curbston_x"], curbstone["curbston_y"]):
            points.extend((float(x), float(y)) for x, y in zip(xs, ys))
        if len(points) < 3:
            return None
        hull = MultiPoint(points).convex_hull
        if hull.is_empty or hull.area <= 0:
            return None
        return hull
    except Exception:
        return None


def _agent_box_polygon(row: Any) -> Polygon:
    return Polygon(_agent_box_corners(row))


def _agent_box_corners(row: Any) -> List[Tuple[float, float]]:
    length = max(float(_row_value(row, "length")), 0.1)
    width = max(float(_row_value(row, "width")), 0.1)
    x = float(_row_value(row, "x"))
    y = float(_row_value(row, "y"))
    heading = float(_row_value(row, "heading"))
    return _box_corners(x, y, heading, length, width)


def _box_corners(
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
) -> List[Tuple[float, float]]:
    local = np.array(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ]
    )
    rot = np.array(
        [
            [math.cos(heading), -math.sin(heading)],
            [math.sin(heading), math.cos(heading)],
        ]
    )
    corners = local @ rot.T + np.array([x, y])
    return [(float(px), float(py)) for px, py in corners]


def _row_value(row: Any, key: str) -> Any:
    if hasattr(row, key):
        return getattr(row, key)
    return row[key]
