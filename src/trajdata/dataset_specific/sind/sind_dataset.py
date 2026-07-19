"""SinD (Signalized Intersections) dataset implementation for trajdata."""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import tqdm

from trajdata.caching.env_cache import EnvCache
from trajdata.caching.scene_cache import SceneCache
from trajdata.data_structures import AgentMetadata, EnvMetadata, Scene, SceneMetadata
from trajdata.data_structures.scene_tag import SceneTag
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.dataset_specific.scene_records import SindSceneRecord
from trajdata.dataset_specific.sind.sind_utils import (
    SIND_LOCATIONS,
    SindObject,
    get_agent_metadata,
    sind_map_to_vector_map,
)
from trajdata.dataset_specific.sind.sind_lanelet2_utils import (
    lanelet2_map_to_vector_map,
    get_lanelet2_map_path,
)
from trajdata.dataset_specific.sind.sind_traffic_lights import (
    build_traffic_light_dataframe,
)
from trajdata.utils import arr_utils

SIND_DATASET_NAME = "sind"


class SindDataset(RawDataset):
    """SinD (Signalized Intersections) dataset implementation for trajdata."""

    def compute_metadata(self, env_name: str, data_dir: str) -> EnvMetadata:
        """Compute metadata for the SinD dataset.

        Args:
            env_name: Name of the environment (must be "sind")
            data_dir: Path to the SinD dataset directory

        Returns:
            EnvMetadata object with dataset information
        """
        if env_name != SIND_DATASET_NAME:
            raise ValueError(f"Unknown SinD env name: {env_name}")

        # Create a temporary object to determine dt
        temp_obj = SindObject(Path(data_dir))
        dt = temp_obj.get_dt()

        return EnvMetadata(
            name=env_name,
            data_dir=data_dir,
            dt=dt,
            parts=[SIND_LOCATIONS],
            scene_split_map=None,
            map_locations=None,
        )

    def load_dataset_obj(self, verbose: bool = False) -> None:
        """Load the SinD dataset object (lazy loading - only loads file paths).

        Args:
            verbose: Whether to print loading information
        """
        if verbose:
            print(f"Loading {self.name} dataset (lazy loading mode)...", flush=True)
        # Lazy loading: only file paths are loaded, data is loaded on-demand
        self.dataset_obj = SindObject(self.metadata.data_dir)
        if verbose:
            print(f"Available locations: {self.dataset_obj.locations}")

        # Track which locations are actually being used (to cache only those)
        self._used_locations: set = set()
        # When loading from cache, we need to infer which locations are being used
        # from the scene_tag that was requested

    def _record_used_location_from_scene_tag(self, scene_tag: SceneTag) -> None:
        """Record which location is being used based on scene_tag."""
        if not hasattr(self, "_used_locations"):
            self._used_locations = set()
        for location in SIND_LOCATIONS:
            if location in scene_tag:
                self._used_locations.add(location)

    def _get_matching_scenes_from_obj(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Union[List[str], None],
        env_cache: EnvCache,
    ) -> List[SceneMetadata]:
        """Compute SceneMetadata for all samples from self.dataset_obj.

        Also saves records to env_cache for later reuse.

        Args:
            scene_tag: Scene tag specifying which scenes to match
            scene_desc_contains: Description filter (not supported for SinD)
            env_cache: Environment cache for storing scene records

        Returns:
            List of SceneMetadata objects
        """
        if scene_desc_contains:
            raise ValueError("SinD dataset does not support scene descriptions.")

        record_list = []
        metadata_list = []

        # Get the location to filter by (if specified)
        # Locations in SinD are: cc, xa, cqNR, tj, cqIR, xasl, cqR
        tag_locations = [s for s in SIND_LOCATIONS if s in scene_tag]

        data_idx = 0
        for location in self.dataset_obj.locations:
            # Skip if tag specifies a different location
            if tag_locations and location not in tag_locations:
                continue

            # Track this location as being used (for selective caching)
            self._used_locations.add(location)

            # Use lazy loading - get scene names for this location
            scene_names = self.dataset_obj._get_scene_names_from_pickle(location)

            for scene_id in scene_names:
                scene_name = f"{location}_{scene_id}"
                scene_length = self.dataset_obj.get_scene_length(scene_name)

                # Skip empty scenes (scenes with no agents or invalid length)
                if scene_length <= 1:
                    continue

                record_list.append(
                    SindSceneRecord(
                        name=scene_name,
                        location=location,
                        length=scene_length,
                        split=location,  # In SinD, each location is its own "split"
                        data_idx=data_idx,
                    )
                )

                metadata_list.append(
                    SceneMetadata(
                        env_name=self.metadata.name,
                        name=scene_name,
                        dt=self.metadata.dt,
                        raw_data_idx=data_idx,
                    )
                )

                data_idx += 1

        self._cache_merged_scenes_list(env_cache, record_list)
        return metadata_list

    def _cache_merged_scenes_list(
        self, env_cache: EnvCache, new_records: List[SindSceneRecord]
    ) -> None:
        """Cache SinD scene records without dropping other locations.

        SinD commonly loads one location at a time to keep memory usage bounded.
        The default RawDataset helper overwrites the environment-level
        scenes_list.dill, which makes later cache-only loads for other locations
        appear empty. Merge by scene name instead.
        """
        merged_records: Dict[str, SindSceneRecord] = {}
        try:
            existing_records: List[SindSceneRecord] = env_cache.load_env_scenes_list(
                self.name
            )
        except FileNotFoundError:
            existing_records = []

        for record in existing_records:
            merged_records[record.name] = record
        for record in new_records:
            merged_records[record.name] = record

        env_cache.save_env_scenes_list(
            self.name,
            sorted(merged_records.values(), key=lambda record: (record.location, record.name)),
        )

    def _get_matching_scenes_from_cache(
        self,
        scene_tag: SceneTag,
        scene_desc_contains: Union[List[str], None],
        env_cache: EnvCache,
    ) -> List[Scene]:
        """Computes Scene data for all samples by reading data from env_cache.

        Args:
            scene_tag: Scene tag specifying which scenes to match
            scene_desc_contains: Description filter (not supported for SinD)
            env_cache: Environment cache for reading scene records

        Returns:
            List of Scene objects
        """
        if scene_desc_contains:
            raise ValueError("SinD dataset does not support scene descriptions.")

        # Record which location is being used (for selective caching)
        self._record_used_location_from_scene_tag(scene_tag)

        # Get the location to filter by (if specified)
        tag_locations = [s for s in SIND_LOCATIONS if s in scene_tag]

        record_list: List[SindSceneRecord] = env_cache.load_env_scenes_list(self.name)

        scenes = []
        for record in record_list:
            # Skip if tag specifies a different location
            if tag_locations and record.location not in tag_locations:
                continue

            # Create Scene object from cached record (like other datasets do)
            scene = Scene(
                env_metadata=self.metadata,
                name=record.name,
                location=record.location,
                data_split=record.split,
                length_timesteps=record.length,
                raw_data_idx=record.data_idx,
                data_access_info=None,  # Not used when loading from cache
            )
            scenes.append(scene)

        return scenes

    def get_scene(self, scene_info: SceneMetadata) -> Scene:
        """Create a Scene object from SceneMetadata.

        Args:
            scene_info: SceneMetadata object

        Returns:
            Scene object
        """
        # Parse scene name to get location
        scene_name = scene_info.name
        location, scene_id = self.dataset_obj._parse_scene_name(scene_name)
        scene_length = self.dataset_obj.get_scene_length(scene_name)

        return self._create_scene(
            scene_name, location, scene_length, scene_info.raw_data_idx
        )

    def _create_scene(
        self, scene_name: str, location: str, length_timesteps: int, data_idx: int
    ) -> Scene:
        """Create a Scene object.

        Args:
            scene_name: Full scene name (e.g., "xa_scene_001")
            location: Location identifier (e.g., "xa")
            length_timesteps: Number of timesteps in the scene
            data_idx: Index into raw dataset

        Returns:
            Scene object
        """
        return Scene(
            env_metadata=self.metadata,
            name=scene_name,
            location=location,
            data_split=location,  # In SinD, each location is its own "split"
            length_timesteps=length_timesteps,
            raw_data_idx=data_idx,
            data_access_info=None,
        )

    def get_agent_info(
        self, scene: Scene, cache_path: Path, cache_class: Type[SceneCache]
    ) -> Tuple[List[AgentMetadata], List[List[AgentMetadata]]]:
        """Get frame-level information from source dataset, caching it to cache_path.

        Always called after cache_maps, can load map if needed to associate
        map information to positions.

        Args:
            scene: Scene object
            cache_path: Path to cache directory
            cache_class: SceneCache class for caching

        Returns:
            Tuple of (agent_list, agent_presence)
                - agent_list: List of AgentMetadata for all agents
                - agent_presence: List of lists, agent_presence[t] contains agents
                  present at timestep t
        """
        scenario = self.dataset_obj.load_scenario(scene.name)

        agent_list: List[AgentMetadata] = []
        agent_presence: List[List[AgentMetadata]] = [[] for _ in range(scene.length_timesteps)]

        df_records = []
        # Track each agent's actual frame range from the State DataFrame
        agent_frame_ranges: Dict[str, Tuple[int, int]] = {}

        # Process each track (trajectory point) in the scene
        tp_info = scenario["tp_info"]

        for tp_id, tp_data in tp_info.items():
            # Create agent metadata
            agent_metadata = get_agent_metadata(str(tp_id), tp_data)
            if agent_metadata is None:
                continue

            # Extract state data from the DataFrame
            state_df = tp_data.get("State")
            if state_df is None or state_df.empty:
                continue

            # Determine the actual frame range from the State DataFrame
            first_frame = int(state_df["frame_id"].min())
            last_frame = int(state_df["frame_id"].max())
            agent_frame_ranges[agent_metadata.name] = (first_frame, last_frame)

            # Update agent metadata with correct timesteps
            agent_metadata.first_timestep = first_frame
            agent_metadata.last_timestep = last_frame

            agent_list.append(agent_metadata)

            # Process each row in the state DataFrame
            for _, row in state_df.iterrows():
                frame_id = int(row["frame_id"])

                # Skip if frame_id is out of range
                if frame_id >= scene.length_timesteps:
                    continue

                # Add agent to presence list
                agent_presence[frame_id].append(agent_metadata)

                # Extract relevant fields
                # SinD columns: track_id, frame_id, timestamp_ms, agent_type,
                # x, y, vx, vy, ax, ay, yaw_rad, heading_rad, length, width

                x = float(row["x"])
                y = float(row["y"])
                vx = float(row.get("vx", 0))
                vy = float(row.get("vy", 0))

                # Use heading if available, otherwise calculate from velocity
                if "heading_rad" in row and pd.notna(row["heading_rad"]):
                    heading = float(row["heading_rad"])
                elif "yaw_rad" in row and pd.notna(row["yaw_rad"]):
                    heading = float(row["yaw_rad"])
                else:
                    # Calculate heading from velocity
                    heading = np.arctan2(vy, vx)

                # Get acceleration if available, otherwise will compute later
                ax = float(row.get("ax", np.nan))
                ay = float(row.get("ay", np.nan))

                # Get size info - use default from metadata if row value is NaN/invalid
                length_val = row.get("length", agent_metadata.extent.length)
                width_val = row.get("width", agent_metadata.extent.width)

                # Check if values are valid (not NaN and positive)
                if pd.isna(length_val) or length_val <= 0:
                    length_val = agent_metadata.extent.length
                if pd.isna(width_val) or width_val <= 0:
                    width_val = agent_metadata.extent.width

                length = float(length_val)
                width = float(width_val)
                height = float(agent_metadata.extent.height)

                df_records.append(
                    {
                        "agent_id": agent_metadata.name,
                        "scene_ts": frame_id,
                        "x": x,
                        "y": y,
                        "z": 0.0,
                        "vx": vx,
                        "vy": vy,
                        "heading": heading,
                        "ax": ax,
                        "ay": ay,
                        "length": length,
                        "width": width,
                        "height": height,
                    }
                )

        # Create and process DataFrame
        df = pd.DataFrame.from_records(df_records)

        if not df.empty:
            df.set_index(["agent_id", "scene_ts"], inplace=True)
            df.sort_index(inplace=True)

            # Compute acceleration if not present or if all NaN
            if "ax" not in df.columns or df["ax"].isna().all():
                df[["ax", "ay"]] = (
                    arr_utils.agent_aware_diff(
                        df[["vx", "vy"]].to_numpy(),
                        df.index.get_level_values(0),
                    )
                    / self.metadata.dt
                )
            else:
                # Fill NaN values with computed acceleration
                accel = arr_utils.agent_aware_diff(
                    df[["vx", "vy"]].to_numpy(),
                    df.index.get_level_values(0),
                ) / self.metadata.dt
                df["ax"] = df["ax"].fillna(pd.Series(accel[:, 0], index=df.index))
                df["ay"] = df["ay"].fillna(pd.Series(accel[:, 1], index=df.index))

            cache_class.save_agent_data(df, cache_path, scene)

        # Optional enhancement: cache SinD traffic-light phase data when the
        # external CSV root is configured. Failures here must not affect the
        # existing trajectory cache path.
        try:
            tls_df, _ = build_traffic_light_dataframe(
                scene_name=scene.name,
                location=scenario["location"],
                scene_id=scenario["scene_id"],
                scene_length=scene.length_timesteps,
                scene_dt=self.metadata.dt,
                pkl_root=Path(self.metadata.data_dir),
            )
            if tls_df is not None:
                cache_class.save_traffic_light_data(tls_df, cache_path, scene)
        except Exception as exc:
            warnings.warn(
                f"Skipping optional SinD traffic-light cache for {scene.name}: {exc}",
                RuntimeWarning,
            )

        return agent_list, agent_presence

    def cache_maps(
        self,
        cache_path: Path,
        map_cache_class: Type[SceneCache],
        map_params: Dict[str, Any],
        scene_tags: Optional[List[SceneTag]] = None,
    ) -> None:
        """Get static, scene-level info from the source dataset, caching it to cache_path.

        Args:
            cache_path: Path to cache directory
            map_cache_class: SceneCache class for caching
            map_params: Dictionary of map parameters (e.g., resolution)
                - use_lanelet2_maps: bool, if True use Lanelet2 format for supported locations
            scene_tags: Optional list of SceneTag objects to determine which locations to cache
        """
        # Check if Lanelet2 maps should be used
        use_lanelet2_maps = map_params.get("use_lanelet2_maps", False)

        # Determine which locations to cache based on scene_tags
        if scene_tags:
            # Extract locations from scene_tags
            locations_to_cache = []
            for tag in scene_tags:
                for location in SIND_LOCATIONS:
                    if location in tag:
                        locations_to_cache.append(location)
                        break
            if locations_to_cache:
                locations_to_cache = list(set(locations_to_cache))  # Remove duplicates
        elif hasattr(self, "_used_locations") and self._used_locations:
            locations_to_cache = list(self._used_locations)
        else:
            # IMPORTANT: By default, only cache the FIRST location to save memory
            # The SinD dataset has 7 locations, each with large pickle files (300-600MB each)
            # Caching all 7 at once would require 3-4 GB of memory
            # Users who want specific locations should use desired_data=["sind-xa"]
            locations_to_cache = [self.dataset_obj.locations[0]]
            print(f"NOTE: Caching only first location '{locations_to_cache[0]}' to save memory.")
            print(f"      To cache other locations, use desired_data=['sind-LOCATION']")

        print(f"Caching maps for {len(locations_to_cache)} location(s): {locations_to_cache}")

        for idx, location in enumerate(locations_to_cache):
            print(f"Loading map for location: {location}")

            # Check if Lanelet2 map should be used for this location
            if use_lanelet2_maps:
                lanelet2_path = get_lanelet2_map_path(location, self.metadata.data_dir)
                if lanelet2_path is not None:
                    print(f"  Using Lanelet2 map: {lanelet2_path}")
                    vector_map = lanelet2_map_to_vector_map(
                        f"{self.name}:{location}", str(lanelet2_path)
                    )
                else:
                    print(f"  Lanelet2 map not found for {location}, using JSON format")
                    # Fall back to JSON format
                    sind_map = self.dataset_obj.load_map(f"{location}_dummy_scene")
                    vector_map = sind_map_to_vector_map(
                        f"{self.name}:{location}", sind_map
                    )
                    # Unload city data after caching
                    self.dataset_obj.unload_city(location)
            else:
                # Use JSON format (default)
                sind_map = self.dataset_obj.load_map(f"{location}_dummy_scene")
                vector_map = sind_map_to_vector_map(
                    f"{self.name}:{location}", sind_map
                )
                # Unload city data after caching
                self.dataset_obj.unload_city(location)

            map_cache_class.finalize_and_cache_map(cache_path, vector_map, map_params)
            print(f"Finished caching {location}")
