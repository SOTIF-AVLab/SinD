"""Reusable SinD scene filtering helpers.

These helpers are intentionally lightweight so downstream scripts can apply the
same basic filters to raw SinD pkl records before building task-specific
examples or mining scenarios.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import KDTree
from shapely.geometry import Point, Polygon


CURBSTONE_PATH = Path(__file__).with_name("data") / "curbstone.json"

Point2D = Tuple[float, float]


@dataclass
class LaneArea:
    """Polygon and reachability metadata for one intersection lane region."""

    polygon: Polygon
    unreachable: List["LaneArea"] = field(default_factory=list)


@dataclass
class LaneAreaResult:
    """Entry and exit lane polygons derived from SinD curbstone key points."""

    in_lanes: List[LaneArea]
    out_lanes: List[LaneArea]


def load_curbstone_points(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the bundled SinD curbstone/key-point data.

    Args:
        path: Optional custom path. Defaults to the packaged
            ``data/curbstone.json`` file.

    Returns:
        Mapping keyed by SinD location id, for example ``cc`` or ``cqR``.
    """
    curbstone_path = CURBSTONE_PATH if path is None else Path(path)
    with curbstone_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def available_curbstone_locations(city_points: Optional[Mapping[str, Any]] = None) -> List[str]:
    """Return sorted SinD locations available in the curbstone key-point file."""
    points = load_curbstone_points() if city_points is None else city_points
    return sorted(points.keys())


def is_vehicle_static(trajectory: Any, threshold: float = 0.01) -> bool:
    """Return ``True`` when a trajectory's start/end displacement is tiny.

    ``trajectory`` is expected to behave like a pandas DataFrame containing
    ``x`` and ``y`` columns.
    """
    if trajectory is None or len(trajectory) == 0:
        return False

    x_diff = abs(trajectory["x"].iloc[-1] - trajectory["x"].iloc[0])
    y_diff = abs(trajectory["y"].iloc[-1] - trajectory["y"].iloc[0])
    return x_diff < threshold and y_diff < threshold


def is_following_vehicle(
    main_vehicle: Mapping[str, Any],
    other_vehicles: Sequence[Mapping[str, Any]],
    lane_area: LaneAreaResult,
    distance_threshold: float = 10.0,
    velocity_threshold: float = 2.0,
    search_radius: float = 50.0,
    max_neighbors: int = 10,
    front_angle_rad: float = math.radians(30.0),
) -> bool:
    """Return ``True`` when the main vehicle appears to be following another.

    Expected vehicle format is the raw SinD pkl frame format used by this
    project: ``{"tp_id": ..., "vehicle_info": {"x", "y", "vx", "vy",
    "heading_rad", ...}}``.
    """
    if not lane_area.in_lanes:
        return False

    main_id = main_vehicle["tp_id"]
    main_info = main_vehicle["vehicle_info"]
    main_pos = (main_info["x"], main_info["y"])
    main_heading = main_info["heading_rad"]

    if not _point_in_any_lane(main_pos, lane_area.in_lanes):
        return False

    candidates = [vehicle for vehicle in other_vehicles if vehicle["tp_id"] != main_id]
    if not candidates:
        return False

    positions = [
        (vehicle["vehicle_info"]["x"], vehicle["vehicle_info"]["y"])
        for vehicle in candidates
    ]
    kdtree = KDTree(positions)
    k = min(len(candidates), max_neighbors)
    distances, indices = kdtree.query(
        main_pos, k=k, distance_upper_bound=search_radius
    )

    distances = np.atleast_1d(distances)
    indices = np.atleast_1d(indices)

    front_vehicle = None
    min_dist = float("inf")
    heading_vec = np.array([math.cos(main_heading), math.sin(main_heading)])

    for idx, dist in zip(indices, distances):
        if not np.isfinite(dist) or idx >= len(candidates):
            continue

        other = candidates[int(idx)]
        other_pos = (other["vehicle_info"]["x"], other["vehicle_info"]["y"])
        rel_vec = np.array(other_pos) - np.array(main_pos)
        rel_dist = np.linalg.norm(rel_vec)
        if rel_dist == 0:
            continue

        cos_theta = np.dot(heading_vec, rel_vec) / rel_dist
        angle = math.acos(np.clip(cos_theta, -1.0, 1.0))
        if angle < front_angle_rad and rel_dist < min_dist:
            min_dist = rel_dist
            front_vehicle = other

    if front_vehicle is None:
        return False

    front_info = front_vehicle["vehicle_info"]
    front_pos = (front_info["x"], front_info["y"])
    if not _point_in_any_lane(front_pos, lane_area.in_lanes):
        return False

    follow_distance = np.linalg.norm(np.array(front_pos) - np.array(main_pos))
    rel_vel = np.linalg.norm(
        [front_info["vx"] - main_info["vx"], front_info["vy"] - main_info["vy"]]
    )

    return follow_distance < distance_threshold and rel_vel < velocity_threshold


def is_position_invalid(
    main_vehicle: Mapping[str, Any],
    outer_path: MplPath,
    lane_area: LaneAreaResult,
    inner_paths: Optional[Iterable[MplPath]] = None,
) -> bool:
    """Return ``True`` when a vehicle is outside drivable area or in an exit lane."""
    pos = (main_vehicle["vehicle_info"]["x"], main_vehicle["vehicle_info"]["y"])

    if not outer_path.contains_point(pos):
        return True

    if inner_paths is not None and any(path.contains_point(pos) for path in inner_paths):
        return True

    return bool(lane_area.out_lanes and _point_in_any_lane(pos, lane_area.out_lanes))


def sort_polygon_points(points: Sequence[Point2D]) -> List[Point2D]:
    """Sort polygon vertices by angle around their centroid."""
    center_x = sum(p[0] for p in points) / len(points)
    center_y = sum(p[1] for p in points) / len(points)

    def angle_from_center(point: Point2D) -> float:
        return math.atan2(point[1] - center_y, point[0] - center_x)

    return sorted(points, key=angle_from_center)


def build_area_boundary(
    city_points: Mapping[str, Any], city: str
) -> Tuple[MplPath, List[MplPath]]:
    """Build drivable-area boundaries from ``curbstone.json`` data."""
    _validate_city(city_points, city)

    x_list = city_points[city]["curbston_x"][0]
    y_list = city_points[city]["curbston_y"][0]

    outer_polygon_points = [(x, y) for x, y in zip(x_list, y_list)]
    outer_path = MplPath(sort_polygon_points(outer_polygon_points))

    inner_paths = []
    for x_inner, y_inner in zip(
        city_points[city]["curbston_x"][1:], city_points[city]["curbston_y"][1:]
    ):
        inner_polygon_points = [(x, y) for x, y in zip(x_inner, y_inner)]
        inner_paths.append(MplPath(sort_polygon_points(inner_polygon_points)))

    return outer_path, inner_paths


def build_lane_area(city_points: Mapping[str, Any], city: str) -> LaneAreaResult:
    """Build entry and exit lane polygons from one city's 24 key points.

    The bundled ``curbstone.json`` records city-specific key points. The same
    point indices are used for all supported SinD intersections to approximate
    entrance and exit lanes.
    """
    _validate_city(city_points, city)

    points = [tuple(point) for point in city_points[city]["key_points"]]
    if len(points) < 24:
        raise ValueError(f"City {city} must contain at least 24 key_points")

    s1 = LaneArea(Polygon([points[19], points[20], points[23], points[22]]))
    s2 = LaneArea(Polygon([points[18], points[19], points[22], points[21]]))
    n2 = LaneArea(Polygon([points[1], points[2], points[5], points[4]]))
    n1 = LaneArea(Polygon([points[0], points[1], points[4], points[3]]))
    w2 = LaneArea(Polygon([points[6], points[7], points[11], points[10]]))
    w1 = LaneArea(Polygon([points[10], points[11], points[15], points[14]]))
    e2 = LaneArea(Polygon([points[12], points[13], points[17], points[16]]))
    e1 = LaneArea(Polygon([points[8], points[9], points[13], points[12]]))

    e_l_1 = [e1]
    e_l_2 = [e2]
    w_l_1 = [w1]
    w_l_2 = [w2]
    s_l_1 = [s1]
    s_l_2 = [s2]
    n_l_1 = [n1]
    n_l_2 = [n2]

    for lane in e_l_1:
        lane.unreachable.extend(e_l_2 + w_l_1 + s_l_1 + n_l_1)
    for lane in n_l_1:
        lane.unreachable.extend(n_l_2 + w_l_1 + s_l_1 + e_l_1)
    for lane in w_l_1:
        lane.unreachable.extend(w_l_2 + n_l_1 + s_l_1 + e_l_1)
    for lane in s_l_1:
        lane.unreachable.extend(s_l_2 + n_l_1 + e_l_1)

    return LaneAreaResult(
        in_lanes=e_l_1 + n_l_1 + w_l_1 + s_l_1,
        out_lanes=e_l_2 + n_l_2 + w_l_2 + s_l_2,
    )


def _point_in_any_lane(pos: Point2D, lanes: Sequence[LaneArea]) -> bool:
    point = Point(pos)
    return any(lane.polygon.contains(point) for lane in lanes)


def _validate_city(city_points: Mapping[str, Any], city: str) -> None:
    if city not in city_points:
        locations = ", ".join(sorted(city_points.keys()))
        raise KeyError(f"Unknown SinD city {city!r}. Available locations: {locations}")
