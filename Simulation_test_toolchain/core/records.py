from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class AgentFrame:
    timestep: int
    agent_name: str
    agent_type: str
    x: float
    y: float
    heading: float
    speed: float
    length: float
    width: float
    is_ego: bool = False
    policy: str = "ground_truth"
    command: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    metadata: Dict[str, Any]
    frames: List[AgentFrame]

    def save_json(self, path: Path) -> None:
        data = {
            "metadata": self.metadata,
            "frames": [asdict(frame) for frame in self.frames],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_csv(self, path: Path) -> None:
        fields = [
            "timestep",
            "agent_name",
            "agent_type",
            "x",
            "y",
            "heading",
            "speed",
            "length",
            "width",
            "is_ego",
            "policy",
            "command",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for frame in self.frames:
                row = asdict(frame)
                row["command"] = json.dumps(row["command"], ensure_ascii=False)
                writer.writerow(row)

