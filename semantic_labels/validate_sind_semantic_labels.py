from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from semantic_labels.labels import DEFAULT_LABEL_PATH, load_semantic_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SinD semantic labels.")
    parser.add_argument("--labels", default=str(DEFAULT_LABEL_PATH))
    parser.add_argument("--data-dir", default="datasets/SinD_dataset")
    parser.add_argument(
        "--check-trajdata",
        action="store_true",
        help="Verify scene_index/source_scene_name against trajdata scene metadata.",
    )
    args = parser.parse_args()

    labels = load_semantic_labels(args.labels)
    errors = validate_basic(labels)
    if args.check_trajdata:
        errors.extend(validate_trajdata(labels, Path(args.data_dir)))
    if errors:
        for error in errors[:50]:
            print(f"ERROR: {error}")
        if len(errors) > 50:
            print(f"... {len(errors) - 50} more errors")
        raise SystemExit(1)

    tag_counts = Counter(tag for label in labels for tag in label["semantic_tags"])
    print(f"Validated {len(labels)} semantic labels")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")


def validate_basic(labels):
    required = {
        "scenario_id",
        "dataset",
        "location",
        "source_scene_name",
        "scene_index",
        "time_window",
        "agents",
        "semantic_tags",
        "semantics",
        "source",
    }
    errors = []
    seen = set()
    for idx, label in enumerate(labels):
        missing = required - set(label)
        if missing:
            errors.append(f"label[{idx}] missing fields: {sorted(missing)}")
            continue
        scenario_id = label["scenario_id"]
        if scenario_id in seen:
            errors.append(f"duplicate scenario_id: {scenario_id}")
        seen.add(scenario_id)
        window = label["time_window"]
        if int(window["end_frame"]) < int(window["start_frame"]):
            errors.append(f"{scenario_id} has negative time window")
        if not label["agents"].get("ego_id"):
            errors.append(f"{scenario_id} missing agents.ego_id")
        if not label["semantic_tags"]:
            errors.append(f"{scenario_id} has no semantic_tags")
    return errors


def validate_trajdata(labels, data_dir: Path):
    from semantic_labels.import_sind_semantic_labels import build_scene_index

    scene_index = build_scene_index(data_dir)
    errors = []
    for label in labels:
        location = label["location"]
        scene_name = label["source_scene_name"]
        meta = scene_index.get(location, {}).get(scene_name)
        if meta is None:
            errors.append(f"{label['scenario_id']} scene not found: {scene_name}")
            continue
        if int(label["scene_index"]) != int(meta["scene_index"]):
            errors.append(f"{label['scenario_id']} scene_index mismatch")
        end_frame = int(label["time_window"]["end_frame"])
        if end_frame >= int(meta["length_timesteps"]):
            errors.append(f"{label['scenario_id']} end_frame outside scene")
    return errors


if __name__ == "__main__":
    main()
