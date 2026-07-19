# Release Manifest

Included:

- SinD 1.0 public sample records under `Data/`.
- SinD 1.0 visualizer under `SIND-Vis-tool/`.
- SinD 2.0 `trajdata` fork source under `src/`.
- Semantic label utilities and sample semantic labels.
- Compact simulation test toolchain source, RiskIDM, Diffusion, minimal example
  configs, and checkpoint path templates.

Excluded:

- full SinD 2.0 PKL data;
- generated `trajdata` caches;
- risk-mining and paper-analysis scripts;
- batch outputs, generated visualizations, and logs;
- model checkpoints, which are provided separately after approval;
- non-released policy implementations, generated model outputs, and third-party
  model repositories.

Before pushing, run:

```bash
find . -type f -size +100M -print
find . -type f \( -name "*.pth" -o -name "*.pth.tar" -o -name "*.pt" \) -print
python -m compileall src semantic_labels Simulation_test_toolchain tools
```
