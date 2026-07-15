from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import dill

from trajdata.dataset_specific.scene_records import SindSceneRecord
from trajdata.dataset_specific.sind.sind_utils import SIND_LOCATIONS, SindObject


def rebuild_scene_list(data_dir: Path, cache_dir: Path) -> None:
    sind = SindObject(data_dir)
    records = []
    data_idx = 0

    for location in SIND_LOCATIONS:
        if location not in sind.locations:
            print(f"{location}: missing")
            continue
        for scene_id in sind._get_scene_names_from_pickle(location):
            scene_name = f"{location}_{scene_id}"
            scene_length = sind.get_scene_length(scene_name)
            if scene_length <= 1:
                continue
            records.append(
                SindSceneRecord(
                    name=scene_name,
                    location=location,
                    length=scene_length,
                    split=location,
                    data_idx=data_idx,
                )
            )
            data_idx += 1
        sind.unload_city(location)

    env_cache_dir = cache_dir / "sind"
    env_cache_dir.mkdir(parents=True, exist_ok=True)
    with (env_cache_dir / "scenes_list.dill").open("wb") as f:
        dill.dump(records, f)

    print(f"wrote {len(records)} records to {env_cache_dir / 'scenes_list.dill'}")
    print(Counter(record.location for record in records))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the global SinD scenes_list.dill for all locations."
    )
    parser.add_argument("--data-dir", default="datasets/SinD_dataset")
    parser.add_argument("--cache-dir", default=str(Path.home() / ".unified_data_cache"))
    args = parser.parse_args()

    rebuild_scene_list(Path(args.data_dir), Path(args.cache_dir))


if __name__ == "__main__":
    main()

