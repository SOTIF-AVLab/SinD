"""Utilities for working with the SinD (Signalized Intersections) dataset."""

import dataclasses
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from trajdata.data_structures.agent import AgentMetadata, AgentType, FixedExtent
from trajdata.maps.vec_map import VectorMap
from trajdata.maps.vec_map_elements import PedWalkway, Polyline, RoadArea, RoadLane

# SinD dataset splits/locations
SIND_LOCATIONS = ("cc", "xa", "cqNR", "tj", "cqIR", "xasl", "cqR")

# Agent type mapping for SinD
# {agent_type: (AgentType, length, width, height)}
# Note: SinD also has broader categories:
#   mv (motor vehicle) = 机动车 - generally maps to VEHICLE
#   nmv (non-motor vehicle) = 非机动车 - generally maps to BICYCLE
#
# Note on tricycle: Chinese tricycles (三轮车) are large vehicles used for cargo
# or passengers, typically 3-4m long and 1-1.7m wide. They are mapped to VEHICLE
# instead of BICYCLE because their size and behavior are more similar to vehicles.
SIND_AGENT_TYPE_DATA: Dict[str, Tuple[AgentType, float, float, float]] = {
    "car": (AgentType.VEHICLE, 4.5, 2.0, 1.5),
    "truck": (AgentType.VEHICLE, 10.0, 2.5, 3.0),
    "bus": (AgentType.VEHICLE, 9.0, 3.0, 4.0),
    "motorcycle": (AgentType.MOTORCYCLE, 2.0, 0.8, 1.5),
    "bicycle": (AgentType.BICYCLE, 1.8, 0.6, 1.5),
    "tricycle": (AgentType.VEHICLE, 3.5, 1.5, 1.5),  # Chinese tricycle = large vehicle
    "pedestrian": (AgentType.PEDESTRIAN, 0.7, 0.7, 1.7),
    # Broader categories (fallbacks)
    "mv": (AgentType.VEHICLE, 4.5, 2.0, 1.5),  # motor vehicle = 机动车
    "nmv": (AgentType.BICYCLE, 1.8, 0.6, 1.5),  # non-motor vehicle = 非机动车
}


@dataclasses.dataclass
class SindCityInfo:
    """Lightweight info about a city (file paths only, no data loaded)."""

    location: str
    tp_info_path: Path
    frame_data_path: Path
    map_path: Path


@dataclasses.dataclass
class SindCityData:
    """Cached data for a single city in SinD dataset."""

    location: str
    tp_info: Dict[str, Any]
    frame_data: Dict[str, Any]
    map_data: Dict[str, Any]

    @property
    def scene_names(self) -> List[str]:
        """Get list of scene IDs for this city."""
        return list(self.tp_info.keys())

    @property
    def num_scenes(self) -> int:
        """Get number of scenes in this city."""
        return len(self.tp_info)


class SindObject:
    """Object for interfacing with SinD data on disk.

    Uses lazy loading to avoid loading all pickle files into memory at once.
    Only loads data for specific cities/scenes when accessed.
    """

    def __init__(
        self, dataset_path: Path, load_locations: Optional[List[str]] = None
    ) -> None:
        """Initialize SindObject with lazy loading.

        Args:
            dataset_path: Path to SinD dataset directory
            load_locations: Optional list of specific locations to load.
                          If None, all available locations will be discovered
                          but not loaded into memory until accessed.
        """
        self.dataset_path = Path(dataset_path)

        # Store file paths only (lazy loading)
        self.city_info: Dict[str, SindCityInfo] = {}
        self._city_data_cache: Dict[str, SindCityData] = {}

        # Discover available locations
        locations_to_check = load_locations if load_locations else SIND_LOCATIONS

        for location in locations_to_check:
            city_path = self.dataset_path / location
            if not city_path.exists():
                continue

            tp_info_path = city_path / f"tp_info_{location}.pkl"
            frame_data_path = city_path / f"frame_data_{location}.pkl"

            # Check for map file: prefer output_json/ directory (correct format)
            # over city directory (may have old flat format)
            output_json_path = self.dataset_path / "output_json" / f"{location}_map.json"
            city_map_path = city_path / f"{location}_map.json"

            # Use output_json version if it exists, otherwise fall back to city directory
            if output_json_path.exists():
                map_path = output_json_path
            elif city_map_path.exists():
                map_path = city_map_path
            else:
                continue

            # Only store paths, don't load data yet
            if tp_info_path.exists() and map_path.exists():
                self.city_info[location] = SindCityInfo(
                    location=location,
                    tp_info_path=tp_info_path,
                    frame_data_path=frame_data_path,
                    map_path=map_path,
                )

    def _load_city_data(self, location: str) -> SindCityData:
        """Load city data from disk (with caching)."""
        if location in self._city_data_cache:
            return self._city_data_cache[location]

        if location not in self.city_info:
            raise ValueError(f"Location {location} not found in dataset")

        info = self.city_info[location]

        try:
            with open(info.tp_info_path, "rb") as f:
                tp_info = pickle.load(f)

            with open(info.frame_data_path, "rb") as f:
                frame_data = pickle.load(f)

            with open(info.map_path, "r") as f:
                map_data = json.load(f)

            city_data = SindCityData(
                location=location,
                tp_info=tp_info,
                frame_data=frame_data,
                map_data=map_data,
            )

            # Cache the loaded data
            self._city_data_cache[location] = city_data
            return city_data

        except FileNotFoundError as e:
            raise ValueError(f"Could not load data for {location}: {e}")

    def _get_scene_names_from_pickle(self, location: str) -> List[str]:
        """Get scene names without loading full pickle into memory.

        Uses a more efficient approach to just read the keys from the pickle.
        """
        if location in self._city_data_cache:
            return self._city_data_cache[location].scene_names

        if location not in self.city_info:
            return []

        # For now, we need to load the pickle to get scene names
        # This is a limitation of pickle format
        city_data = self._load_city_data(location)
        return city_data.scene_names

    @property
    def scenario_names(self) -> List[str]:
        """Get all scene names across all locations."""
        result = []
        for location in self.city_info.keys():
            scene_names = self._get_scene_names_from_pickle(location)
            for scene_name in scene_names:
                result.append(f"{location}_{scene_name}")
        return result

    @property
    def locations(self) -> List[str]:
        """Get list of available locations."""
        return list(self.city_info.keys())

    def _parse_scene_name(self, scene_name: str) -> Tuple[str, str]:
        """Parse scene name into location and scene_id.

        Args:
            scene_name: Format "{location}_{scene_id}"

        Returns:
            Tuple of (location, scene_id)
        """
        if "_" in scene_name:
            parts = scene_name.split("_", 1)
            if parts[0] in self.city_info:
                return parts[0], parts[1]

        # Try to find location by checking loaded cities
        for location in self._city_data_cache.keys():
            city_data = self._city_data_cache[location]
            if scene_name in city_data.tp_info:
                return location, scene_name

        # If not found in cache, check all cities
        for location in self.city_info.keys():
            city_data = self._load_city_data(location)
            if scene_name in city_data.tp_info:
                return location, scene_name

        raise ValueError(f"Could not parse scene name: {scene_name}")

    def get_city_data(self, scene_name: str) -> SindCityData:
        """Get city data for a given scene name (lazy loads if needed)."""
        location, _ = self._parse_scene_name(scene_name)
        return self._load_city_data(location)

    def get_scene_id(self, scene_name: str) -> str:
        """Get scene_id from full scene name."""
        location, scene_id = self._parse_scene_name(scene_name)
        return scene_id

    def load_scenario(self, scene_name: str) -> Dict[str, Any]:
        """Load scenario data for a given scene."""
        city_data = self.get_city_data(scene_name)
        scene_id = self.get_scene_id(scene_name)

        if scene_id not in city_data.tp_info:
            raise ValueError(f"Scene {scene_id} not found in location {city_data.location}")

        return {
            "location": city_data.location,
            "scene_id": scene_id,
            "tp_info": city_data.tp_info[scene_id],
            "frame_data": city_data.frame_data.get(scene_id, {}),
        }

    def load_map(self, scene_name: str) -> Dict[str, Any]:
        """Load map data for a given scene."""
        city_data = self.get_city_data(scene_name)
        return city_data.map_data

    def get_dt(self) -> float:
        """Calculate the timestep (dt) from the data.

        SinD is typically 10Hz (0.1s), but actual timestamps may have small variations.
        """
        for location in self.city_info.keys():
            scene_names = self._get_scene_names_from_pickle(location)
            if scene_names:
                city_data = self._load_city_data(location)
                first_scene = scene_names[0]
                tp_info = city_data.tp_info[first_scene]

                # Look at the first track's state data
                for tp_id, tp_data in tp_info.items():
                    if "State" in tp_data:
                        state_df = tp_data["State"]
                        if "timestamp_ms" in state_df.columns:
                            timestamps = state_df["timestamp_ms"].values
                            if len(timestamps) > 1:
                                # Calculate mean difference in seconds
                                dt_ms = np.mean(np.diff(timestamps))
                                dt = dt_ms / 1000.0
                                # Round to nearest 0.01 to handle floating point variations
                                # SinD is nominally 10Hz (0.1s)
                                dt_rounded = round(dt, 2)
                                # Snap to common values (0.1, 0.05, 0.02, etc.)
                                if abs(dt_rounded - 0.1) < 0.02:
                                    return 0.1
                                elif abs(dt_rounded - 0.05) < 0.01:
                                    return 0.05
                                else:
                                    return dt_rounded

        # Default fallback if we can't determine dt
        return 0.1

    def get_scene_length(self, scene_name: str) -> int:
        """Get the length of a scene in timesteps."""
        city_data = self.get_city_data(scene_name)
        scene_id = self.get_scene_id(scene_name)

        tp_info = city_data.tp_info[scene_id]

        # Find the max frame_id across all tracks
        max_frame = 0
        for tp_id, tp_data in tp_info.items():
            if "State" in tp_data:
                state_df = tp_data["State"]
                if "frame_id" in state_df.columns:
                    max_frame = max(max_frame, state_df["frame_id"].max())

        return int(max_frame) + 1

    def unload_city(self, location: str) -> None:
        """Unload a city's data from memory to free up space.

        Args:
            location: Location identifier (e.g., "xa", "cc")
        """
        if location in self._city_data_cache:
            del self._city_data_cache[location]

    def unload_all(self) -> None:
        """Unload all cached city data from memory."""
        self._city_data_cache.clear()


def sind_agent_type_mapping(agent_type: str) -> Optional[Tuple[AgentType, FixedExtent]]:
    """Map SinD agent type to trajdata AgentType and extent.

    Args:
        agent_type: SinD agent type string

    Returns:
        Tuple of (AgentType, FixedExtent) or None if unknown type
    """
    agent_data = SIND_AGENT_TYPE_DATA.get(agent_type)
    if agent_data is None:
        return None

    agent_type_enum, length, width, height = agent_data
    extent = FixedExtent(length=length, width=width, height=height)

    return agent_type_enum, extent


def get_agent_metadata(
    agent_id: str, tp_data: Dict[str, Any]
) -> Optional[AgentMetadata]:
    """Create AgentMetadata from SinD trajectory data.

    Args:
        agent_id: Agent identifier
        tp_data: Trajectory-point data from SinD

    Returns:
        AgentMetadata or None if agent type is unknown
    """
    # In SinD dataset, "Class" contains the actual vehicle type (car, bus, etc.)
    # while "Type" is a broader category (mv = motor vehicle, etc.)
    agent_type_str = tp_data.get("Class", tp_data.get("Type", ""))
    agent_mapping = sind_agent_type_mapping(agent_type_str)

    if agent_mapping is None:
        return None

    agent_type_enum, extent = agent_mapping

    # Use size from data if available and valid, otherwise use default
    if "Length" in tp_data and "Width" in tp_data:
        try:
            length = float(tp_data["Length"])
            width = float(tp_data["Width"])
            # Check if values are valid (not NaN or infinite)
            if not (np.isnan(length) or np.isnan(width) or
                    np.isinf(length) or np.isinf(width) or
                    length <= 0 or width <= 0):
                # Use default height from mapping
                _, _, _, height = SIND_AGENT_TYPE_DATA.get(
                    agent_type_str, (AgentType.VEHICLE, 4.5, 2.0, 1.5)
                )
                extent = FixedExtent(length=length, width=width, height=height)
            # else: fall back to default extent from mapping
        except (ValueError, TypeError):
            # If conversion fails, use default extent from mapping
            pass

    # IMPORTANT: Do NOT use InitialFrame/FinalFrame from tp_data directly.
    # These values in SinD data are not reliable (FinalFrame can be a weird float).
    # Instead, we'll determine first_timestep and last_timestep from the actual
    # frame_id values in the State DataFrame. This is done in get_agent_info()
    # in sind_dataset.py, which will update the agent metadata after reading
    # the actual state data. Here we set placeholder values that will be updated.
    first_frame = 0
    last_frame = 0

    return AgentMetadata(
        name=agent_id,
        agent_type=agent_type_enum,
        first_timestep=first_frame,
        last_timestep=last_frame,
        extent=extent,
    )


def sind_map_to_vector_map(map_id: str, sind_map: Dict[str, Any]) -> VectorMap:
    """Convert SinD map data to trajdata VectorMap format.

    SinD maps have the following structure (from output_json/ directory):
    - pedestrian_area: List of polylines (each is a list of [x, y] points)
    - drivable_area: List of polylines (each is a list of [x, y] points)
    - road_divider: List of polylines (each is a list of [x, y] points)
    - lane_divider: List of polylines (each is a list of [x, y] points)

    Note: Despite the name "drivable_area" and "pedestrian_area", these are
    polylines (lines), not filled polygons. The drivable_area contains curbstone
    lines that define the boundary of drivable areas.

    Args:
        map_id: Map identifier (format: "env_name:map_name")
        sind_map: SinD map dictionary

    Returns:
        VectorMap populated with SinD map elements
    """
    vector_map = VectorMap(map_id)

    extents_min = None
    extents_max = None

    # Process drivable areas (curbstone lines)
    # Each polyline is closed to form a polygon representing a drivable area
    if "drivable_area" in sind_map:
        for idx, polyline_points in enumerate(sind_map["drivable_area"]):
            if not polyline_points or len(polyline_points) < 3:
                continue

            area_arr = np.array(polyline_points, dtype=np.float64)

            # Handle different input formats
            if area_arr.ndim == 1:
                # Single point
                area_arr = area_arr.reshape(1, -1)
            if area_arr.ndim != 2 or area_arr.shape[0] < 3:
                # Need at least 3 points for a polygon
                continue
            if area_arr.shape[1] == 2:
                # Add z=0 if only 2D points
                area_arr = np.column_stack([area_arr, np.zeros(len(area_arr))])
            elif area_arr.shape[1] != 3:
                # Skip if not 2D or 3D
                continue

            # Close the polyline to form a polygon if not already closed
            if not np.allclose(area_arr[0], area_arr[-1]):
                area_arr = np.vstack([area_arr, area_arr[0]])

            # Update extents
            if extents_min is None:
                extents_min = area_arr.min(0)
                extents_max = area_arr.max(0)
            else:
                extents_min = np.minimum(extents_min, area_arr.min(0))
                extents_max = np.maximum(extents_max, area_arr.max(0))

            vector_map.add_map_element(
                RoadArea(
                    id=f"RoadArea_{idx}",
                    exterior_polygon=Polyline(area_arr),
                )
            )

    # Process pedestrian areas
    # Each polyline is closed to form a polygon representing a pedestrian walkway
    if "pedestrian_area" in sind_map:
        for idx, polyline_points in enumerate(sind_map["pedestrian_area"]):
            if not polyline_points or len(polyline_points) < 3:
                continue

            area_arr = np.array(polyline_points, dtype=np.float64)

            # Handle different input formats
            if area_arr.ndim == 1:
                # Single point
                area_arr = area_arr.reshape(1, -1)
            if area_arr.ndim != 2 or area_arr.shape[0] < 3:
                # Need at least 3 points for a polygon
                continue
            if area_arr.shape[1] == 2:
                # Add z=0 if only 2D points
                area_arr = np.column_stack([area_arr, np.zeros(len(area_arr))])
            elif area_arr.shape[1] != 3:
                # Skip if not 2D or 3D
                continue

            # Close the polyline to form a polygon if not already closed
            if not np.allclose(area_arr[0], area_arr[-1]):
                area_arr = np.vstack([area_arr, area_arr[0]])

            # Update extents
            if extents_min is None:
                extents_min = area_arr.min(0)
                extents_max = area_arr.max(0)
            else:
                extents_min = np.minimum(extents_min, area_arr.min(0))
                extents_max = np.maximum(extents_max, area_arr.max(0))

            vector_map.add_map_element(
                PedWalkway(
                    id=f"PedWalkway_{idx}",
                    polygon=Polyline(area_arr),
                )
            )

    # Process road_divider and lane_divider as RoadLanes
    # These are polylines (lines), not polygons
    lane_idx = 0
    for divider_name, divider_key in [("road_divider", "road_divider"), ("lane_divider", "lane_divider")]:
        if divider_key in sind_map:
            for idx, polyline_points in enumerate(sind_map[divider_key]):
                if not polyline_points:
                    continue

                line_arr = np.array(polyline_points, dtype=np.float64)

                # Handle different input formats
                if line_arr.ndim == 1:
                    line_arr = line_arr.reshape(1, -1)
                if line_arr.shape[1] == 2:
                    line_arr = np.column_stack([line_arr, np.zeros(len(line_arr))])
                elif line_arr.shape[1] != 3:
                    continue

                # Update extents
                if extents_min is None:
                    extents_min = line_arr.min(0)
                    extents_max = line_arr.max(0)
                else:
                    extents_min = np.minimum(extents_min, line_arr.min(0))
                    extents_max = np.maximum(extents_max, line_arr.max(0))

                # Create a simple lane from the divider line
                # Add heading if not present
                if line_arr.shape[1] == 3:
                    # Compute heading from successive points
                    headings = np.arctan2(
                        np.diff(line_arr[:, 1], prepend=line_arr[0, 1]),
                        np.diff(line_arr[:, 0], prepend=line_arr[0, 0])
                    )
                    line_arr_with_h = np.column_stack([line_arr, headings])
                else:
                    line_arr_with_h = line_arr

                vector_map.add_map_element(
                    RoadLane(
                        id=f"Lane_{divider_name}_{idx}",
                        center=Polyline(line_arr_with_h),
                        left_edge=None,
                        right_edge=None,
                        adj_lanes_left=set(),
                        adj_lanes_right=set(),
                        next_lanes=set(),
                        prev_lanes=set(),
                    )
                )
                lane_idx += 1

    # Set map extent
    if extents_min is not None and extents_max is not None:
        # extent is [min_x, min_y, min_z, max_x, max_y, max_z]
        vector_map.extent = np.concatenate([extents_min, extents_max])
    else:
        # Default extent if no areas found
        vector_map.extent = np.array([0, 0, 0, 100, 100, 5])

    return vector_map
