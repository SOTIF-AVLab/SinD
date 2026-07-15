from __future__ import annotations

import argparse
from pathlib import Path

from Simulation_test_toolchain.core.config import load_config, make_project_dir
from Simulation_test_toolchain.core.runner import run_simulation
from Simulation_test_toolchain.visualization.bokeh_result import save_interactive_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SinD simulation test project.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_dir = make_project_dir(cfg, args.config)
    result = run_simulation(cfg)

    if cfg.output.save_json:
        result.save_json(project_dir / "trajectory_log.json")
    if cfg.output.save_csv:
        result.save_csv(project_dir / "trajectory_log.csv")
    if cfg.visualization.enabled and cfg.output.save_html:
        save_interactive_html(result, project_dir / "interactive.html", cfg)

    print(f"Simulation test complete: {project_dir}")
    print(f"HTML: {project_dir / 'interactive.html'}")


if __name__ == "__main__":
    main()

