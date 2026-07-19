from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class PolicyState:
    agent_name: str
    dt: float = 0.1
    initialized: bool = False


@dataclass
class PolicyAction:
    xyh: np.ndarray
    command: Dict[str, Any] = field(default_factory=dict)


class BasePolicy(ABC):
    policy_name = "base"

    def __init__(self, dt: float = 0.1, **_: Any) -> None:
        self.dt = dt
        self.state: Optional[PolicyState] = None
        self.last_command: Dict[str, Any] = {}

    @abstractmethod
    def reset(self, obs, ego_idx: int = 0) -> PolicyState:
        raise NotImplementedError

    @abstractmethod
    def get_action(self, obs, ego_idx: int = 0) -> PolicyAction:
        raise NotImplementedError

