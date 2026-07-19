from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Div, HoverTool, Slider
from bokeh.models.tools import WheelZoomTool
from bokeh.plotting import figure

from Simulation_test_toolchain.core.records import SimulationResult


def save_interactive_html(result: SimulationResult, path: Path, cfg) -> None:
    frames = result.frames
    if not frames:
        raise ValueError("No frames available for visualization.")

    timesteps = sorted({frame.timestep for frame in frames})
    first_t = timesteps[0]
    xs = [frame.x for frame in frames]
    ys = [frame.y for frame in frames]
    margin = cfg.visualization.map_margin_m

    fig = figure(
        width=cfg.visualization.width,
        height=cfg.visualization.height,
        match_aspect=True,
        x_range=(min(xs) - margin, max(xs) + margin),
        y_range=(min(ys) - margin, max(ys) + margin),
        title=(
            f"SinD {result.metadata['location']} scene {result.metadata['scene_index']} "
            f"| ego={result.metadata['ego_agent']} | policy={result.metadata['ego_policy']}"
        ),
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    _apply_default_settings(fig)
    _draw_static_map_if_available(fig, result, cfg)

    all_data = _build_frame_data(frames)
    initial = _slice_data(all_data, first_t)
    source = ColumnDataSource(initial)
    full_source = ColumnDataSource(all_data)

    history_source = ColumnDataSource(_build_history_data(frames, first_t))
    fig.multi_line(
        xs="xs",
        ys="ys",
        line_color="color",
        line_alpha=0.55,
        line_width=2,
        source=history_source,
    )

    rects = fig.patches(
        xs="rect_xs",
        ys="rect_ys",
        fill_color="color",
        fill_alpha="alpha",
        line_color="line_color",
        line_width="line_width",
        source=source,
    )
    fig.patches(
        xs="dir_xs",
        ys="dir_ys",
        fill_color="color",
        fill_alpha="alpha",
        line_color="line_color",
        line_width="line_width",
        source=source,
    )
    fig.add_tools(
        HoverTool(
            renderers=[rects],
            tooltips=[
                ("agent", "@agent_name"),
                ("ego", "@is_ego"),
                ("policy", "@policy"),
                ("speed", "@speed{0.00} m/s"),
                ("pos", "(@x{0.00}, @y{0.00})"),
            ],
        )
    )

    command_div = Div(text=_command_html(frames, first_t), width=420)
    slider = Slider(
        start=timesteps[0],
        end=timesteps[-1],
        step=1,
        value=first_t,
        title="Scene timestep",
        width=cfg.visualization.width - 80,
    )
    slider.js_on_change(
        "value",
        CustomJS(
            args=dict(
                source=source,
                full_source=full_source,
                history_source=history_source,
                command_div=command_div,
            ),
            code="""
const t = cb_obj.value;
const full = full_source.data;
const next = {};
for (const key in source.data) { next[key] = []; }
for (let i = 0; i < full['timestep'].length; i++) {
  if (full['timestep'][i] === t) {
    for (const key in next) { next[key].push(full[key][i]); }
  }
}
source.data = next;

const names = [...new Set(full['agent_name'])];
const hist = {xs: [], ys: [], color: []};
for (const name of names) {
  const hx = [];
  const hy = [];
  let color = '#999999';
  for (let i = 0; i < full['timestep'].length; i++) {
    if (full['agent_name'][i] === name && full['timestep'][i] <= t) {
      hx.push(full['x'][i]);
      hy.push(full['y'][i]);
      color = full['color'][i];
    }
  }
  if (hx.length > 0) {
    hist.xs.push(hx);
    hist.ys.push(hy);
    hist.color.push(color);
  }
}
history_source.data = hist;

let egoText = '<b>Ego command</b><br>No ego frame at this timestep.';
for (let i = 0; i < full['timestep'].length; i++) {
  if (full['timestep'][i] === t && full['is_ego'][i] === true) {
    egoText = '<b>Ego command</b><br>' +
      'timestep: ' + t + '<br>' +
      'agent: ' + full['agent_name'][i] + '<br>' +
      'policy: ' + full['policy'][i] + '<br>' +
      '<pre>' + full['command_json'][i] + '</pre>';
    break;
  }
}
command_div.text = egoText;
""",
        ),
    )

    output_file(path, title="SinD Simulation Test")
    save(column(fig, slider, row(command_div)))


def _build_frame_data(frames) -> Dict[str, List]:
    data = {
        "timestep": [],
        "agent_name": [],
        "agent_type": [],
        "x": [],
        "y": [],
        "heading": [],
        "speed": [],
        "is_ego": [],
        "policy": [],
        "command_json": [],
        "rect_xs": [],
        "rect_ys": [],
        "dir_xs": [],
        "dir_ys": [],
        "color": [],
        "alpha": [],
        "line_color": [],
        "line_width": [],
    }
    for frame in frames:
        rect, direction = _compute_agent_rect_coords(frame.heading, frame.length, frame.width)
        color = "#d62728" if frame.is_ego else "#1f77b4"
        data["timestep"].append(frame.timestep)
        data["agent_name"].append(frame.agent_name)
        data["agent_type"].append(frame.agent_type)
        data["x"].append(frame.x)
        data["y"].append(frame.y)
        data["heading"].append(frame.heading)
        data["speed"].append(frame.speed)
        data["is_ego"].append(frame.is_ego)
        data["policy"].append(frame.policy)
        data["command_json"].append(json.dumps(frame.command, indent=2))
        data["rect_xs"].append((rect[:, 0] + frame.x).tolist())
        data["rect_ys"].append((rect[:, 1] + frame.y).tolist())
        data["dir_xs"].append((direction[:, 0] + frame.x).tolist())
        data["dir_ys"].append((direction[:, 1] + frame.y).tolist())
        data["color"].append(color)
        data["alpha"].append(0.85 if frame.is_ego else 0.55)
        data["line_color"].append("black" if frame.is_ego else "#333333")
        data["line_width"].append(2.0 if frame.is_ego else 0.7)
    return data


def _slice_data(data: Dict[str, List], timestep: int) -> Dict[str, List]:
    out = {key: [] for key in data}
    for idx, t in enumerate(data["timestep"]):
        if t == timestep:
            for key in out:
                out[key].append(data[key][idx])
    return out


def _build_history_data(frames, timestep: int) -> Dict[str, List]:
    by_agent: Dict[str, Dict[str, List]] = {}
    for frame in frames:
        if frame.timestep > timestep:
            continue
        entry = by_agent.setdefault(
            frame.agent_name,
            {"xs": [], "ys": [], "color": "#d62728" if frame.is_ego else "#1f77b4"},
        )
        entry["xs"].append(frame.x)
        entry["ys"].append(frame.y)
    return {
        "xs": [entry["xs"] for entry in by_agent.values()],
        "ys": [entry["ys"] for entry in by_agent.values()],
        "color": [entry["color"] for entry in by_agent.values()],
    }


def _command_html(frames, timestep: int) -> str:
    for frame in frames:
        if frame.timestep == timestep and frame.is_ego:
            return (
                "<b>Ego command</b><br>"
                f"timestep: {timestep}<br>"
                f"agent: {frame.agent_name}<br>"
                f"policy: {frame.policy}<br>"
                f"<pre>{json.dumps(frame.command, indent=2)}</pre>"
            )
    return "<b>Ego command</b><br>No ego frame at this timestep."


def _draw_static_map_if_available(fig, result: SimulationResult, cfg) -> None:
    map_name = result.metadata.get("map_name")
    cache_path = result.metadata.get("cache_path")
    if not map_name or not cache_path:
        return
    try:
        from trajdata.maps.map_api import MapAPI
        from trajdata.utils import vis_utils

        xs = [frame.x for frame in result.frames]
        ys = [frame.y for frame in result.frames]
        bbox = (
            min(xs) - cfg.visualization.map_margin_m,
            max(xs) + cfg.visualization.map_margin_m,
            min(ys) - cfg.visualization.map_margin_m,
            max(ys) + cfg.visualization.map_margin_m,
        )
        map_api = MapAPI(Path(cache_path))
        vec_map = map_api.get_map(
            map_name,
            incl_road_lanes=True,
            incl_road_areas=True,
            incl_ped_crosswalks=True,
            incl_ped_walkways=True,
        )
        vis_utils.draw_map_elems(fig, vec_map, np.eye(3), bbox=bbox)
    except Exception as exc:
        fig.title.text += f" | map unavailable: {exc}"


def _apply_default_settings(fig) -> None:
    fig.match_aspect = True
    fig.grid.visible = False
    fig.toolbar.active_scroll = fig.select_one(WheelZoomTool)
    fig.toolbar.autohide = True
    fig.xaxis.axis_label_text_font_size = "10pt"
    fig.xaxis.major_label_text_font_size = "10pt"
    fig.yaxis.axis_label_text_font_size = "10pt"
    fig.yaxis.major_label_text_font_size = "10pt"
    fig.title.text_font_size = "13pt"


def _compute_agent_rect_coords(
    heading: float, length: float, width: float
) -> tuple[np.ndarray, np.ndarray]:
    base_rect = np.array(
        [
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
            [length / 2, width / 2],
            [length / 2, -width / 2],
        ]
    )
    base_dir = np.array(
        [
            [0, np.sqrt(3) / 3],
            [-1 / 2, -np.sqrt(3) / 6],
            [1 / 2, -np.sqrt(3) / 6],
        ]
    )
    rect = _rotate(base_rect, heading)
    direction = _rotate(base_dir, heading - np.pi / 2)
    return rect, direction


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return points @ rot.T
