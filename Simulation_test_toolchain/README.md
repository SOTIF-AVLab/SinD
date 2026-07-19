# Simulation Test Toolchain

This folder contains the compact SinD 2.0 scenario testing toolchain. The public
release keeps only the basic runner, RiskIDM, and Diffusion policy template.
Other internal policy evaluation code is intentionally not included.

## Run RiskIDM

```bash
conda activate sind2
python -m Simulation_test_toolchain.run_test \
  --config Simulation_test_toolchain/test_projects/example_sind_single/config.yaml
```

## Run Diffusion

```bash
conda activate sind2
python -m Simulation_test_toolchain.run_test \
  --config Simulation_test_toolchain/test_projects/example_diffuser/config.yaml
```

The Diffusion checkpoint is not distributed through GitHub. After an approved
application, place the provided checkpoint at:

```text
Simulation_test_toolchain/checkpoints/v2.0.ckpt
```

## Outputs

Outputs are written under `Simulation_test_toolchain/test_projects/<project_name>/`:

- `config.yaml`: copied run configuration;
- `trajectory_log.json`: per-timestep agent states and ego commands;
- `trajectory_log.csv`: flat trajectory log;
- `interactive.html`: Bokeh timeline with map, agents, ego highlight, and ego command panel.

Generated outputs are ignored by git.

## Policies

Supported policy names:

- `ground_truth`: replay the next state from trajdata future data;
- `risk_idm`: built-in risk-aware IDM controller;
- `diffuser`: Diffusion wrapper using `checkpoints.diffuser_ckpt_path`.

The active Python environment must be internally consistent. In particular,
`trajdata` imports `pandas`, `pyarrow`, and `torch`; a NumPy 2.x environment
with extensions compiled against NumPy 1.x or a mismatched CUDA PyTorch install
will fail before simulation starts.

## Semantic Label Workflow

Configs may refer to a semantic label id:

```yaml
scenario:
  semantic_label_id: SIND_TJ_MPRTTC_8_3_4_R4_00001
  semantic_label_path: datasets/SinD_dataset/Semantic_labels/scenarios.json
  semantic_min_num_steps: 150
```

When `semantic_label_id` is set, the runner resolves `dataset.location`,
`scenario.scene_index`, `scenario.init_timestep`, and `scenario.ego_agent_name`
from the label. `scenario.num_steps` is expanded to cover the label window and
at least `semantic_min_num_steps`.

Validate labels against trajdata scene metadata:

```bash
python -m semantic_labels.validate_sind_semantic_labels \
  --labels datasets/SinD_dataset/Semantic_labels/scenarios.json \
  --data-dir datasets/SinD_dataset \
  --check-trajdata
```
