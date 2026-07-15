#!/usr/bin/env python3
"""Build a self-contained Bokeh HTML viewer for SinD semantic scenario clips."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Div, HoverTool, Select, Slider, TextInput
from bokeh.models.tools import WheelZoomTool
from bokeh.plotting import figure


AGENT_COLORS = {
    "car": "#1f77b4",
    "truck": "#9467bd",
    "bus": "#8c564b",
    "motorcycle": "#ff7f0e",
    "bicycle": "#2ca02c",
    "tricycle": "#17becf",
    "pedestrian": "#d62728",
    "mv": "#1f77b4",
    "nmv": "#2ca02c",
    "unknown": "#7f7f7f",
}


def load_labels(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("scenarios", payload) if isinstance(payload, dict) else payload
    if not isinstance(labels, list):
        raise ValueError(f"Semantic labels must be a list or contain scenarios: {path}")
    return labels


def scene_key_from_label(label: Dict[str, Any]) -> str:
    location = str(label["location"])
    scene_name = str(label.get("source_scene_name") or "")
    prefix = f"{location}_"
    if scene_name.startswith(prefix):
        return scene_name[len(prefix) :]
    return scene_name


def load_location_tp(data_dir: Path, location: str) -> Dict[str, Any]:
    path = data_dir / location / f"tp_info_{location}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory PKL: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def load_map(data_dir: Path, location: str) -> Dict[str, Any]:
    candidates = [
        data_dir / "output_json" / f"{location}_map.json",
        data_dir / location / f"{location}_map.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def choose_labels(
    labels: List[Dict[str, Any]],
    *,
    max_scenarios: int,
    tags: Optional[Iterable[str]],
) -> List[Dict[str, Any]]:
    tag_filter = set(tags or [])
    chosen: List[Dict[str, Any]] = []
    seen_ids = set()

    # Prefer a balanced first pass across tags so the viewer is useful out of the box.
    if not tag_filter:
        seen_tags = set()
        for label in labels:
            label_tags = list(label.get("semantic_tags", []))
            if not label_tags:
                continue
            if label_tags[0] in seen_tags:
                continue
            chosen.append(label)
            seen_ids.add(label["scenario_id"])
            seen_tags.add(label_tags[0])
            if len(chosen) >= max_scenarios:
                return chosen

    for label in labels:
        label_tags = set(label.get("semantic_tags", []))
        if tag_filter and not label_tags.intersection(tag_filter):
            continue
        scenario_id = label.get("scenario_id")
        if scenario_id in seen_ids:
            continue
        chosen.append(label)
        seen_ids.add(scenario_id)
        if len(chosen) >= max_scenarios:
            break
    return chosen


def build_clip_payload(
    label: Dict[str, Any],
    scene_tracks: Dict[str, Dict[str, Any]],
    map_data: Dict[str, Any],
    *,
    context_frames: int,
    max_agents: int,
) -> Dict[str, Any]:
    window = label.get("time_window", {})
    start = max(0, int(window.get("start_frame", 0)) - context_frames)
    end = int(window.get("end_frame", start)) + context_frames
    ego_id = str(label.get("agents", {}).get("ego_id", ""))

    frame_rows: List[Dict[str, Any]] = []
    agents_seen: Dict[str, int] = {}
    for agent_id, track in scene_tracks.items():
        state = track.get("State")
        if state is None or state.empty or "frame_id" not in state.columns:
            continue
        df = state[(state["frame_id"] >= start) & (state["frame_id"] <= end)].copy()
        if df.empty:
            continue
        agent_key = str(agent_id)
        overlap = int(len(df))
        agents_seen[agent_key] = overlap

    keep_agents = sorted(
        agents_seen,
        key=lambda aid: (aid != ego_id, -agents_seen[aid], aid),
    )[:max_agents]
    keep_set = set(keep_agents)

    for agent_id, track in scene_tracks.items():
        agent_key = str(agent_id)
        if agent_key not in keep_set:
            continue
        state = track["State"]
        df = state[(state["frame_id"] >= start) & (state["frame_id"] <= end)].copy()
        if df.empty:
            continue
        agent_class = str(track.get("Class") or df.get("agent_type", pd.Series(["unknown"])).iloc[0]).lower()
        agent_type = str(track.get("Type") or agent_class).lower()
        length = float(track.get("Length") or df.get("length", pd.Series([4.5])).iloc[0] or 4.5)
        width = float(track.get("Width") or df.get("width", pd.Series([1.8])).iloc[0] or 1.8)
        for row in df.to_dict(orient="records"):
            x = float(row.get("x", np.nan))
            y = float(row.get("y", np.nan))
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            heading = row.get("heading_rad", row.get("yaw_rad", 0.0))
            try:
                heading = float(heading)
            except Exception:
                heading = 0.0
            if not np.isfinite(heading):
                heading = 0.0
            vx = float(row.get("vx", 0.0) or 0.0)
            vy = float(row.get("vy", 0.0) or 0.0)
            rect, direction = agent_patch(x, y, heading, length, width, agent_class)
            is_ego = agent_key == ego_id
            color = "#e11d48" if is_ego else AGENT_COLORS.get(agent_class, AGENT_COLORS.get(agent_type, "#64748b"))
            frame_rows.append(
                {
                    "frame": int(row["frame_id"]),
                    "agent_id": agent_key,
                    "agent_type": agent_class,
                    "x": x,
                    "y": y,
                    "heading": heading,
                    "speed": float(math.hypot(vx, vy)),
                    "is_ego": is_ego,
                    "rect_xs": rect[:, 0].tolist(),
                    "rect_ys": rect[:, 1].tolist(),
                    "dir_xs": direction[:, 0].tolist(),
                    "dir_ys": direction[:, 1].tolist(),
                    "color": color,
                    "alpha": 0.9 if is_ego else 0.58,
                    "line_color": "#111827" if is_ego else "#475569",
                    "line_width": 2.0 if is_ego else 0.7,
                }
            )

    if not frame_rows:
        raise ValueError(f"No trajectory rows for {label.get('scenario_id')}")

    frames = sorted({row["frame"] for row in frame_rows})
    xs = [row["x"] for row in frame_rows]
    ys = [row["y"] for row in frame_rows]
    margin = 12.0
    return {
        "label": compact_label(label),
        "frames": frames,
        "rows": frame_rows,
        "history": build_history(frame_rows),
        "map": map_to_payload(map_data),
        "x_range": [float(min(xs) - margin), float(max(xs) + margin)],
        "y_range": [float(min(ys) - margin), float(max(ys) + margin)],
    }


def compact_label(label: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scenario_id": label.get("scenario_id"),
        "location": label.get("location"),
        "source_scene_name": label.get("source_scene_name"),
        "scene_index": label.get("scene_index"),
        "semantic_tags": label.get("semantic_tags", []),
        "time_window": label.get("time_window", {}),
        "agents": label.get("agents", {}),
        "semantics": label.get("semantics", {}),
    }


def agent_patch(
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
    agent_class: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if agent_class == "pedestrian":
        length = max(min(length, 0.9), 0.5)
        width = max(min(width, 0.9), 0.5)
    rect = np.array(
        [
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ],
        dtype=float,
    )
    tri = np.array(
        [
            [length / 2, 0.0],
            [length / 2 - min(length * 0.35, 1.2), width * 0.35],
            [length / 2 - min(length * 0.35, 1.2), -width * 0.35],
        ],
        dtype=float,
    )
    rot = np.array(
        [[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]]
    )
    offset = np.array([x, y])
    return rect @ rot.T + offset, tri @ rot.T + offset


def build_history(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: (item["agent_id"], item["frame"])):
        by_agent.setdefault(row["agent_id"], []).append(row)
    history = []
    for agent_id, agent_rows in by_agent.items():
        history.append(
            {
                "agent_id": agent_id,
                "frames": [row["frame"] for row in agent_rows],
                "xs": [row["x"] for row in agent_rows],
                "ys": [row["y"] for row in agent_rows],
                "color": agent_rows[0]["color"],
            }
        )
    return history


def map_to_payload(map_data: Dict[str, Any]) -> Dict[str, List]:
    payload = {"xs": [], "ys": [], "color": [], "alpha": [], "width": []}
    styles = {
        "drivable_area": ("#94a3b8", 0.45, 1.0),
        "pedestrian_area": ("#16a34a", 0.35, 1.0),
        "road_divider": ("#475569", 0.6, 1.0),
        "lane_divider": ("#64748b", 0.5, 0.8),
    }
    for key, (color, alpha, width) in styles.items():
        for line in map_data.get(key, []) or []:
            points = np.asarray(line, dtype=float)
            if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
                continue
            payload["xs"].append(points[:, 0].tolist())
            payload["ys"].append(points[:, 1].tolist())
            payload["color"].append(color)
            payload["alpha"].append(alpha)
            payload["width"].append(width)
    return payload


def build_sources(clips: List[Dict[str, Any]]) -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List]]:
    labels = {
        "scenario_id": [],
        "location": [],
        "scene": [],
        "tags": [],
        "start": [],
        "end": [],
        "ego": [],
        "summary": [],
    }
    rows = {
        "clip": [],
        "frame": [],
        "agent_id": [],
        "agent_type": [],
        "x": [],
        "y": [],
        "heading": [],
        "speed": [],
        "is_ego": [],
        "rect_xs": [],
        "rect_ys": [],
        "dir_xs": [],
        "dir_ys": [],
        "color": [],
        "alpha": [],
        "line_color": [],
        "line_width": [],
    }
    histories = {"clip": [], "agent_id": [], "frames": [], "xs": [], "ys": [], "color": []}
    for idx, clip in enumerate(clips):
        label = clip["label"]
        window = label.get("time_window", {})
        labels["scenario_id"].append(label.get("scenario_id", ""))
        labels["location"].append(label.get("location", ""))
        labels["scene"].append(label.get("source_scene_name", ""))
        labels["tags"].append(",".join(label.get("semantic_tags", [])))
        labels["start"].append(int(window.get("start_frame", clip["frames"][0])))
        labels["end"].append(int(window.get("end_frame", clip["frames"][-1])))
        labels["ego"].append(str(label.get("agents", {}).get("ego_id", "")))
        labels["summary"].append(json.dumps(label, ensure_ascii=False, indent=2))
        for row in clip["rows"]:
            rows["clip"].append(idx)
            for key in rows:
                if key != "clip":
                    rows[key].append(row[key])
        for history in clip["history"]:
            histories["clip"].append(idx)
            for key in ["agent_id", "frames", "xs", "ys", "color"]:
                histories[key].append(history[key])
    return labels, rows, histories


def build_viewer(clips: List[Dict[str, Any]], output: Path) -> None:
    labels, all_rows, all_histories = build_sources(clips)
    initial_clip = 0
    initial_frame = clips[0]["frames"][0]
    initial_rows = slice_rows(all_rows, initial_clip, initial_frame)
    initial_histories = slice_histories(all_histories, initial_clip, initial_frame)
    initial_map = clips[0]["map"]

    fig = figure(
        width=1120,
        height=760,
        match_aspect=True,
        x_range=clips[0]["x_range"],
        y_range=clips[0]["y_range"],
        title="SinD semantic scenario clip",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        output_backend="canvas",
    )
    fig.grid.visible = False
    fig.axis.visible = False
    wheel = fig.select_one(WheelZoomTool)
    if wheel is not None:
        fig.toolbar.active_scroll = wheel

    map_source = ColumnDataSource(initial_map)
    fig.multi_line(
        xs="xs",
        ys="ys",
        line_color="color",
        line_alpha="alpha",
        line_width="width",
        source=map_source,
    )

    history_source = ColumnDataSource(initial_histories)
    fig.multi_line(
        xs="xs",
        ys="ys",
        line_color="color",
        line_alpha=0.55,
        line_width=2,
        source=history_source,
    )

    agent_source = ColumnDataSource(initial_rows)
    rects = fig.patches(
        xs="rect_xs",
        ys="rect_ys",
        fill_color="color",
        fill_alpha="alpha",
        line_color="line_color",
        line_width="line_width",
        source=agent_source,
    )
    fig.patches(
        xs="dir_xs",
        ys="dir_ys",
        fill_color="color",
        fill_alpha="alpha",
        line_color="line_color",
        line_width="line_width",
        source=agent_source,
    )
    fig.add_tools(
        HoverTool(
            renderers=[rects],
            tooltips=[
                ("agent", "@agent_id"),
                ("type", "@agent_type"),
                ("ego", "@is_ego"),
                ("speed", "@speed{0.00} m/s"),
                ("frame", "@frame"),
            ],
        )
    )

    label_source = ColumnDataSource(labels)
    row_source = ColumnDataSource(all_rows)
    history_full_source = ColumnDataSource(all_histories)
    clip_meta_source = ColumnDataSource(
        {
            "x0": [clip["x_range"][0] for clip in clips],
            "x1": [clip["x_range"][1] for clip in clips],
            "y0": [clip["y_range"][0] for clip in clips],
            "y1": [clip["y_range"][1] for clip in clips],
            "frames": [clip["frames"] for clip in clips],
            "map_xs": [clip["map"]["xs"] for clip in clips],
            "map_ys": [clip["map"]["ys"] for clip in clips],
            "map_color": [clip["map"]["color"] for clip in clips],
            "map_alpha": [clip["map"]["alpha"] for clip in clips],
            "map_width": [clip["map"]["width"] for clip in clips],
        }
    )

    tag_values = sorted({tag for tags in labels["tags"] for tag in tags.split(",") if tag})
    scene_values = sorted(set(labels["scene"]))
    scenario_values = labels["scenario_id"]
    tag_select = Select(title="Semantic tag", value="All", options=["All"] + tag_values, width=260)
    scene_select = Select(title="Scene", value="All", options=["All"] + scene_values, width=260)
    id_select = Select(title="Scenario id", value=scenario_values[0], options=scenario_values, width=440)
    id_search = TextInput(title="Search id / ego", value="", width=260)
    slider = Slider(
        start=clips[0]["frames"][0],
        end=clips[0]["frames"][-1],
        step=1,
        value=initial_frame,
        title="Frame",
        width=920,
    )
    info = Div(text="", width=440)

    callback = CustomJS(
        args=dict(
            labels=label_source,
            rows=row_source,
            histories=history_full_source,
            meta=clip_meta_source,
            agent_source=agent_source,
            history_source=history_source,
            map_source=map_source,
            tag_select=tag_select,
            scene_select=scene_select,
            id_select=id_select,
            id_search=id_search,
            slider=slider,
            info=info,
            fig=fig,
        ),
        code=VIEWER_JS,
    )
    for widget in [tag_select, scene_select, id_select, id_search, slider]:
        widget.js_on_change("value", callback)
    info.text = "<b>Scenario</b><br>" + labels["summary"][0].replace("\n", "<br>")

    output_file(output, title="SinD Semantic Scenario Video Preview")
    save(column(row(tag_select, scene_select, id_search), id_select, slider, row(fig, info)))


def slice_rows(all_rows: Dict[str, List], clip: int, frame: int) -> Dict[str, List]:
    out = {key: [] for key in all_rows if key != "clip"}
    for idx, row_clip in enumerate(all_rows["clip"]):
        if row_clip == clip and all_rows["frame"][idx] == frame:
            for key in out:
                out[key].append(all_rows[key][idx])
    return out


def slice_histories(all_histories: Dict[str, List], clip: int, frame: int) -> Dict[str, List]:
    out = {"xs": [], "ys": [], "color": []}
    for idx, row_clip in enumerate(all_histories["clip"]):
        if row_clip != clip:
            continue
        frames = all_histories["frames"][idx]
        xs = all_histories["xs"][idx]
        ys = all_histories["ys"][idx]
        keep_x = [x for x, t in zip(xs, frames) if t <= frame]
        keep_y = [y for y, t in zip(ys, frames) if t <= frame]
        if keep_x:
            out["xs"].append(keep_x)
            out["ys"].append(keep_y)
            out["color"].append(all_histories["color"][idx])
    return out


VIEWER_JS = """
function matchingIndices() {
  const data = labels.data;
  const tag = tag_select.value;
  const scene = scene_select.value;
  const query = id_search.value.toLowerCase().trim();
  const out = [];
  for (let i = 0; i < data.scenario_id.length; i++) {
    if (tag !== 'All' && !data.tags[i].split(',').includes(tag)) continue;
    if (scene !== 'All' && data.scene[i] !== scene) continue;
    if (query && !(data.scenario_id[i].toLowerCase().includes(query) || String(data.ego[i]).toLowerCase().includes(query))) continue;
    out.push(i);
  }
  return out;
}

function refreshScenarioOptions(matches) {
  const old = id_select.value;
  const opts = matches.map(i => labels.data.scenario_id[i]);
  id_select.options = opts.length ? opts : [''];
  if (!opts.includes(old)) {
    id_select.value = opts.length ? opts[0] : '';
  }
}

let matches = matchingIndices();
if (cb_obj !== id_select && cb_obj !== slider) {
  refreshScenarioOptions(matches);
}
let clip = labels.data.scenario_id.indexOf(id_select.value);
if (clip < 0 && matches.length > 0) clip = matches[0];
if (clip < 0) return;

const frames = meta.data.frames[clip];
if (slider.start !== frames[0] || slider.end !== frames[frames.length - 1]) {
  slider.start = frames[0];
  slider.end = frames[frames.length - 1];
  slider.value = frames[0];
}
let frame = slider.value;
if (!frames.includes(frame)) {
  frame = frames[0];
  slider.value = frame;
}

const rd = rows.data;
const next = {};
for (const key in agent_source.data) next[key] = [];
for (let i = 0; i < rd.clip.length; i++) {
  if (rd.clip[i] === clip && rd.frame[i] === frame) {
    for (const key in next) next[key].push(rd[key][i]);
  }
}
agent_source.data = next;

const hd = histories.data;
const hist = {xs: [], ys: [], color: []};
for (let i = 0; i < hd.clip.length; i++) {
  if (hd.clip[i] !== clip) continue;
  const hx = [];
  const hy = [];
  for (let j = 0; j < hd.frames[i].length; j++) {
    if (hd.frames[i][j] <= frame) {
      hx.push(hd.xs[i][j]);
      hy.push(hd.ys[i][j]);
    }
  }
  if (hx.length > 0) {
    hist.xs.push(hx);
    hist.ys.push(hy);
    hist.color.push(hd.color[i]);
  }
}
history_source.data = hist;

map_source.data = {
  xs: meta.data.map_xs[clip],
  ys: meta.data.map_ys[clip],
  color: meta.data.map_color[clip],
  alpha: meta.data.map_alpha[clip],
  width: meta.data.map_width[clip],
};
fig.x_range.start = meta.data.x0[clip];
fig.x_range.end = meta.data.x1[clip];
fig.y_range.start = meta.data.y0[clip];
fig.y_range.end = meta.data.y1[clip];
fig.title.text = labels.data.scenario_id[clip] + ' | frame ' + frame;
info.text = '<b>Scenario</b><br><pre>' + labels.data.summary[clip] + '</pre>';
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("datasets/SinD_dataset/Semantic_labels/scenarios_sample.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/SinD_dataset"))
    parser.add_argument("--output", type=Path, default=Path("semantic_preview/semantic_video_preview.html"))
    parser.add_argument("--max-scenarios", type=int, default=24)
    parser.add_argument("--max-agents", type=int, default=40)
    parser.add_argument("--context-frames", type=int, default=20)
    parser.add_argument("--tags", nargs="*", default=None)
    args = parser.parse_args()

    labels = choose_labels(load_labels(args.labels), max_scenarios=args.max_scenarios, tags=args.tags)
    if not labels:
        raise SystemExit("No semantic labels selected.")

    tp_cache: Dict[str, Dict[str, Any]] = {}
    map_cache: Dict[str, Dict[str, Any]] = {}
    clips: List[Dict[str, Any]] = []
    for label in labels:
        location = str(label["location"])
        if location not in tp_cache:
            try:
                tp_cache[location] = load_location_tp(args.data_dir, location)
            except FileNotFoundError as exc:
                print(f"skip {label.get('scenario_id')}: {exc}")
                continue
        map_cache.setdefault(location, load_map(args.data_dir, location))
        scene_key = scene_key_from_label(label)
        scene_tracks = tp_cache[location].get(scene_key)
        if scene_tracks is None:
            print(f"skip {label.get('scenario_id')}: scene not found {location}/{scene_key}")
            continue
        try:
            clips.append(
                build_clip_payload(
                    label,
                    scene_tracks,
                    map_cache[location],
                    context_frames=args.context_frames,
                    max_agents=args.max_agents,
                )
            )
        except Exception as exc:
            print(f"skip {label.get('scenario_id')}: {exc}")
        if len(clips) >= args.max_scenarios:
            break

    if not clips:
        raise SystemExit("No clips could be built from selected labels.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_viewer(clips, args.output)
    print(f"Wrote {args.output} with {len(clips)} clips")


if __name__ == "__main__":
    main()
