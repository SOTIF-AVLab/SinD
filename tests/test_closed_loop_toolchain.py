from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from Simulation_test_toolchain.core.batch_metrics import _compute_risk_metrics
from Simulation_test_toolchain.core.config import SimulationConfig, ToolchainConfig
from Simulation_test_toolchain.core.records import AgentFrame
from Simulation_test_toolchain.core.runner import _select_controlled_non_ego_indices


def _obs(names, poses):
    return SimpleNamespace(
        agent_name=names,
        agents_from_world_tf=np.stack(
            [_agent_from_world(x, y, h) for x, y, h, _, _ in poses], axis=0
        ),
        agent_hist=np.asarray(
            [
                [[0.0, 0.0], [speed * 0.1, 0.0]]
                for _, _, _, speed, _ in poses
            ],
            dtype=np.float32,
        ),
        agent_hist_len=np.asarray([2] * len(poses), dtype=np.int64),
        agent_hist_extent=np.asarray(
            [[[length, 1.8, 1.5]] * 2 for _, _, _, _, length in poses],
            dtype=np.float32,
        ),
        dt=np.asarray([0.1] * len(poses), dtype=np.float32),
    )


def _agent_from_world(x, y, heading):
    world_from_agent = np.eye(3, dtype=np.float32)
    c, s = np.cos(heading), np.sin(heading)
    world_from_agent[:2, :2] = [[c, -s], [s, c]]
    world_from_agent[:2, 2] = [x, y]
    return np.linalg.inv(world_from_agent)


def test_select_controlled_non_ego_front_nearest_limit():
    obs = _obs(
        ["ego", "near", "far", "behind", "stopped", "side"],
        [
            (0.0, 0.0, 0.0, 2.0, 4.2),
            (5.0, 0.0, 0.0, 1.0, 4.2),
            (8.0, 0.0, 0.0, 1.0, 4.2),
            (-2.0, 0.0, 0.0, 1.0, 4.2),
            (3.0, 0.0, 0.0, 0.0, 4.2),
            (4.0, 3.0, 0.0, 1.0, 4.2),
        ],
    )
    cfg = ToolchainConfig(
        simulation=SimulationConfig(
            mode="multi_agent_closed_loop",
            controlled_neighbor_radius_m=10.0,
            controlled_neighbor_max_agents=2,
            controlled_neighbor_forward_only=True,
            controlled_neighbor_min_speed_mps=0.3,
        )
    )

    selected = _select_controlled_non_ego_indices(obs, 0, cfg)

    selected_names = {obs.agent_name[idx] for idx in selected}
    assert selected_names == {"near", "side"}
    assert "behind" not in selected_names
    assert "stopped" not in selected_names


def test_risk_metrics_detect_collision_course():
    frames = [
        AgentFrame(0, "ego", "VEHICLE", 0.0, 0.0, 0.0, 2.0, 4.0, 2.0, True),
        AgentFrame(0, "other", "VEHICLE", 8.0, 0.0, np.pi, 2.0, 4.0, 2.0, False),
        AgentFrame(1, "ego", "VEHICLE", 0.2, 0.0, 0.0, 2.0, 4.0, 2.0, True),
        AgentFrame(1, "other", "VEHICLE", 7.8, 0.0, np.pi, 2.0, 4.0, 2.0, False),
    ]
    df = pd.DataFrame.from_records([frame.__dict__ for frame in frames])

    metrics = _compute_risk_metrics(df, dt=0.1)

    assert metrics["MRD"] > 0.0
    assert metrics["ARD"] >= metrics["MRD"]
    assert metrics["MinTTC"] < 2.5
    assert metrics["AveTTC"] <= 2.5
