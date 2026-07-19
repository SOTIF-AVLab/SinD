"""Lanelet2-style OSM map parsing utilities for SinD dataset.

This module provides utilities to parse Lanelet2-style OSM files for SinD dataset.
Since the SinD OSM files don't strictly conform to Lanelet2 library requirements,
we use direct XML parsing instead of the lanelet2 library.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial import ConvexHull

import numpy as np
import pyproj
import tqdm

from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import RoadLane, Polyline, PedCrosswalk, MapElementType


# 设置地图中心点的纬度和经度
orgins = {'xa': [34.3825639017, 108.88751600454],'cc': [0,0], 'cqNR': [0,0], 'tj': [0,0], 'cqIR': [0,0], 'xasl': [0,0], 'cqR': [0,0]}

class LL2XYProjector:
    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin + 180.) / 6) + 1  # works for most tiles, and for all in the dataset
        self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]

def lanelet2_map_to_vector_map(map_id: str, map_path: str) -> VectorMap:
    """Convert Lanelet2-style OSM map to trajdata VectorMap.

    This function parses a Lanelet2-style OSM file using direct XML parsing
    (instead of the lanelet2 library) and converts it to a trajdata VectorMap.
    The SinD OSM files have lanelets with single borders which don't conform
    to the Lanelet2 library's strict requirements.

    Args:
        map_id: Map identifier (format: "env_name:map_name")
        map_path: Path to Lanelet2 OSM file

    Returns:
        VectorMap populated with lane elements (with basic connectivity)
    """
    # Parse the OSM XML file
    tree = ET.parse(map_path)
    root = tree.getroot()

    # Build dictionaries for nodes, ways, and relations
    nodes: Dict[int, Tuple[float, float, float]] = {}
    ways: Dict[int, List[int]] = {}
    way_tags: Dict[int, Dict[str, str]] = {}  # Store way tags
    lanelets: Dict[int, Dict[str, Optional[int]]] = {}  # lanelet_id -> {left, right, name}

    # Parse nodes (assuming lat/lon are small UTM offsets, convert to meters)
    # Using the same conversion as sind_map.py: lat * 111000 + offset_x, lon * 111000 + offset_y
    for node in root.findall("node"):
        node_id = int(node.get("id"))
        lat = float(node.get("lat"))
        lon = float(node.get("lon"))

        # Convert lat/lon to meters (assuming local UTM-like coordinates)
        
        xy = LL2XYProjector(orgins[map_id.split(":")[1]][0], orgins[map_id.split(":")[1]][1]).latlon2xy(lat, lon)
        x, y = xy
        z = 0.0  # Assume z=0 for all nodes
        nodes[node_id] = (x, y, z)

    # Parse ways
    for way in root.findall("way"):
        way_id = int(way.get("id"))
        node_refs = [int(nd.get("ref")) for nd in way.findall("nd")]
        ways[way_id] = node_refs
        # Store way tags for later zebra_marking processing
        tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
        way_tags[way_id] = tags

    # Parse lanelets (relations with type='lanelet')
    crosswalks: Dict[int, Dict[str, Optional[int]]] = {}  # For crosswalk lanelets
    for relation in root.findall("relation"):
        tags = {tag.get("k"): tag.get("v") for tag in relation.findall("tag")}
        if tags.get("type") != "lanelet":
            continue

        relation_id = int(relation.get("id"))
        left_way = None
        right_way = None
        name = tags.get("name", f"lanelet_{relation_id}")

        for member in relation.findall("member"):
            role = member.get("role")
            ref = int(member.get("ref"))
            if role == "left":
                left_way = ref
            elif role == "right":
                right_way = ref

        # Check if this is a crosswalk (by subtype)
        if tags.get("subtype") == "crosswalk":
            crosswalks[relation_id] = {
                "left": left_way,
                "right": right_way,
                "name": name,
            }
        else:
            # Regular lanelet (road lane)
            lanelets[relation_id] = {
                "left": left_way,
                "right": right_way,
                "name": name,
            }

    # Create VectorMap
    vector_map = VectorMap(map_id)

    # Track map extent
    maximum_bound: np.ndarray = np.full((3,), -np.inf)
    minimum_bound: np.ndarray = np.full((3,), np.inf)

    # Build lane connectivity from shared borders
    # Lanelets that share a border way are connected
    way_to_lanelets: Dict[int, List[int]] = {}
    for lanelet_id, lanelet_data in lanelets.items():
        if lanelet_data["left"]:
            way_to_lanelets.setdefault(lanelet_data["left"], []).append(lanelet_id)
        if lanelet_data["right"]:
            way_to_lanelets.setdefault(lanelet_data["right"], []).append(lanelet_id)

    # Create RoadLanes from lanelets
    for lanelet_id, lanelet_data in tqdm.tqdm(
        lanelets.items(), desc=f"Loading {map_id} lanes", leave=False
    ):
        left_way_id = lanelet_data["left"]
        right_way_id = lanelet_data["right"]

        # Skip if no border defined
        if left_way_id is None and right_way_id is None:
            continue

        # Extract boundary points
        left_pts = None
        right_pts = None

        if left_way_id is not None and left_way_id in ways:
            left_pts = np.array([nodes[nid] for nid in ways[left_way_id] if nid in nodes])
        if right_way_id is not None and right_way_id in ways:
            right_pts = np.array([nodes[nid] for nid in ways[right_way_id] if nid in nodes])

        # Compute centerline as average of left and right boundaries
        if left_pts is not None and right_pts is not None:
            # Interpolate to get centerline
            min_len = min(len(left_pts), len(right_pts))
            center_pts = (left_pts[:min_len] + right_pts[:min_len]) / 2
        elif left_pts is not None:
            center_pts = left_pts.copy()
        elif right_pts is not None:
            center_pts = right_pts.copy()
        else:
            continue

        # Create RoadLane
        road_lane = RoadLane(
            id=lanelet_data["name"],
            center=Polyline(center_pts),
            left_edge=Polyline(left_pts) if left_pts is not None else None,
            right_edge=Polyline(right_pts) if right_pts is not None else None,
            adj_lanes_left=set(),
            adj_lanes_right=set(),
            next_lanes=set(),
            prev_lanes=set(),
        )

        # Fill basic connectivity from shared borders
        _fill_lane_connectivity_from_borders(road_lane, lanelet_id, lanelet_data, way_to_lanelets, lanelets)

        # Add to vector map
        vector_map.add_map_element(road_lane)

        # Update map extent
        for pts in [center_pts, left_pts, right_pts]:
            if pts is not None:
                maximum_bound = np.maximum(maximum_bound, pts.max(axis=0))
                minimum_bound = np.minimum(minimum_bound, pts.min(axis=0))

    # Parse zebra_marking ways to create PedCrosswalk elements
    # Crosswalks are lanelet relations with subtype='crosswalk'
    # Each crosswalk has left and right zebra_marking way boundaries
    for cw_id, cw_data in crosswalks.items():
        left_way_id = cw_data["left"]
        right_way_id = cw_data["right"]

        # Skip if no boundary defined
        if left_way_id is None or right_way_id is None:
            continue

        # Extract boundary points from the two ways
        left_pts = None
        right_pts = None

        if left_way_id is not None and left_way_id in ways:
            left_pts = np.array([nodes[nid] for nid in ways[left_way_id] if nid in nodes])
        if right_way_id is not None and right_way_id in ways:
            right_pts = np.array([nodes[nid] for nid in ways[right_way_id] if nid in nodes])

        if left_pts is None or right_pts is None:
            continue

        # Combine points from both boundaries
        # Handle lines that might be in opposite directions
        combined_pts = np.vstack([left_pts, right_pts])

        # Create convex hull from the combined points
        try:
            hull = ConvexHull(combined_pts[:, :2])  # Use only x, y for convex hull
            hull_indices = hull.vertices

            # Get the hull points (including z coordinate)
            hull_pts = combined_pts[hull_indices]

            # Close the polygon if not already closed
            if not np.allclose(hull_pts[0], hull_pts[-1]):
                hull_pts = np.vstack([hull_pts, hull_pts[0]])

            # Create PedCrosswalk
            ped_crosswalk = PedCrosswalk(
                id=cw_data["name"],
                polygon=Polyline(hull_pts),
            )
            vector_map.add_map_element(ped_crosswalk)

            # Update map extent
            maximum_bound = np.maximum(maximum_bound, hull_pts.max(axis=0))
            minimum_bound = np.minimum(minimum_bound, hull_pts.min(axis=0))
        except Exception as e:
            # Convex hull may fail if points are collinear or insufficient
            print(f"Warning: Could not create convex hull for crosswalk {cw_data['name']}: {e}")
            continue

    # Set map extent
    # vector_map.extent is [min_x, min_y, min_z, max_x, max_y, max_z]
    if np.isfinite(minimum_bound).all() and np.isfinite(maximum_bound).all():
        vector_map.extent = np.concatenate((minimum_bound, maximum_bound))
    else:
        # Default extent if no lanes found
        vector_map.extent = np.array([0, 0, 0, 100, 100, 5])

    return vector_map


def _fill_lane_connectivity_from_borders(
    road_lane: RoadLane,
    lanelet_id: int,
    lanelet_data: Dict[str, Optional[int]],
    way_to_lanelets: Dict[int, List[int]],
    lanelets: Dict[int, Dict[str, Optional[int]]],
) -> None:
    """Fill lane connectivity relationships from shared border ways.

    This is a simplified connectivity approach that doesn't use the Lanelet2
    RoutingGraph. It infers connectivity from shared borders between lanelets.

    Args:
        road_lane: RoadLane object to populate with connectivity
        lanelet_id: ID of the current lanelet
        lanelet_data: Dictionary with left/right way IDs for this lanelet
        way_to_lanelets: Mapping from way ID to list of lanelet IDs using it
        lanelets: Dictionary of all lanelets
    """
    # Find adjacent lanes (sharing left or right border)
    left_way_id = lanelet_data["left"]
    right_way_id = lanelet_data["right"]

    # Check adjacent lanelets sharing the same border
    if left_way_id in way_to_lanelets:
        for other_id in way_to_lanelets[left_way_id]:
            if other_id != lanelet_id:
                # Determine if this is left or right adjacency based on border usage
                other_data = lanelets[other_id]
                if other_data["right"] == left_way_id:
                    road_lane.adj_lanes_left.add(lanelets[other_id]["name"])
                elif other_data["left"] == left_way_id:
                    road_lane.adj_lanes_right.add(lanelets[other_id]["name"])

    if right_way_id in way_to_lanelets:
        for other_id in way_to_lanelets[right_way_id]:
            if other_id != lanelet_id:
                other_data = lanelets[other_id]
                if other_data["left"] == right_way_id:
                    road_lane.adj_lanes_right.add(lanelets[other_id]["name"])
                elif other_data["right"] == right_way_id:
                    road_lane.adj_lanes_left.add(lanelets[other_id]["name"])

    # For next/prev connectivity, we'd need more sophisticated analysis
    # of lane endpoints. For now, leave them empty.


def get_lanelet2_map_path(location: str, dataset_dir: Union[str, Path]) -> Optional[Path]:
    """Get the Lanelet2-style OSM map file path for a location (if it exists).

    Args:
        location: Location identifier (e.g., "tj", "cqNR")
        dataset_dir: Path to SinD dataset directory

    Returns:
        Path to OSM file if it exists, None otherwise
    """
    # All locations potentially have Lanelet2 maps
    # The code will check if the file exists

    # Check for lanelet2 map in the dataset directory's Lanelet_maps_SinD subdirectory
    data_path = Path(dataset_dir)
    lanelet2_dir = data_path / "Lanelet_maps_SinD"
    osm_file = lanelet2_dir / f"lanelet2_{location}.osm"

    if osm_file.exists():
        return osm_file

    # Fallback: check in repository root (for backwards compatibility)
    repo_root = Path(__file__).resolve().parents[4]  # Go up to repo root
    lanelet2_dir = repo_root / "Lanelet2_map_SinD"
    osm_file = lanelet2_dir / f"lanelet2_{location}.osm"

    if osm_file.exists():
        return osm_file

    return None
