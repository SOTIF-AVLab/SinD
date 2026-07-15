from __future__ import annotations

import random
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class EgoSelectionStrategy(str, Enum):
    LONGEST_TRAJECTORY = "longest_trajectory"
    FASTEST = "fastest"
    RANDOM = "random"
    FIRST = "first"


def select_ego_from_scene(
    scene,
    strategy: str = EgoSelectionStrategy.LONGEST_TRAJECTORY,
    ego_agent_name: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[object, int]:
    if not scene.agents:
        raise ValueError(f"Scene {scene.name} has no agents.")

    if ego_agent_name is not None:
        for idx, agent in enumerate(scene.agents):
            if agent.name == ego_agent_name:
                return agent, idx
        raise ValueError(f"Requested ego_agent_name={ego_agent_name!r} not in scene.")

    strategy = EgoSelectionStrategy(strategy)
    if strategy == EgoSelectionStrategy.FIRST:
        selected_idx = 0
    elif strategy == EgoSelectionStrategy.RANDOM:
        rng = random.Random(seed)
        selected_idx = rng.randrange(len(scene.agents))
    elif strategy == EgoSelectionStrategy.FASTEST:
        # Without loading full tracks here, duration is a stable proxy.
        selected_idx = int(
            np.argmax(
                [
                    agent.last_timestep - agent.first_timestep
                    for agent in scene.agents
                ]
            )
        )
    else:
        selected_idx = int(
            np.argmax(
                [
                    agent.last_timestep - agent.first_timestep
                    for agent in scene.agents
                ]
            )
        )

    return scene.agents[selected_idx], selected_idx

