# SST RL Framework

Reusable GRPO post-training infrastructure for simultaneous speech translation
and other speech-domain RL tasks.

This repository is a framework-oriented refactor built upon the HPO-style
simultaneous translation RL code path from "Hierarchical Policy Optimization for
Simultaneous Translation of Unbounded Speech" (ACL 2026,
[arXiv:2604.21045](https://arxiv.org/pdf/2604.21045)). The paper is cited as
the method and environment foundation; this repository focuses on refactoring
that paper-specific RL path into a cleaner, reusable framework for future
speech research.

For SFT and the original InfiniSST data/model pipeline, refer to
[InfiniSST](https://github.com/LeiLiLab/InfiniSST). This repository focuses on
RL post-training.

## What This Repo Provides

- A reusable GRPO post-training stack for speech tasks.
- Stable protocols for rollout, environment stepping, rewards, and task plugins.
- Backward-compatible HPO/InfiniSST training entrypoints.
- A modular InfiniSST task plugin that can be used as a template for new tasks.
- Config inheritance for shared GRPO, language, and reward-model settings.

## Why This Refactor Exists

The original RL code path was tightly coupled to one simultaneous translation
environment. That made it hard to reuse the infrastructure for nearby speech RL
work such as retrieval-aware SST, speech agents, streaming ASR correction, or
task-specific reward-model experiments.

This refactor separates the framework from the task:

- The framework owns rollout collection, environment calls, reward aggregation,
  GRPO optimization, logging, and checkpointing.
- Task plugins own data construction, task metadata, Ray environment setup, and
  domain-specific reward dependencies.

The result is still compatible with existing InfiniSST/HPO configs, but new
speech RL tasks should plug into the framework through task specs instead of
editing the GRPO trainer directly.

## Architecture

| Layer | Package | Responsibility |
| --- | --- | --- |
| Task API | `nemo_rl.posttraining` | Typed protocols, environment adapters, `PostTrainingTaskSpec`, task registry |
| Rollouts | `nemo_rl.experience` | Generation, environment stepping, sync/async rollout runners, rollout metrics |
| Algorithms | `nemo_rl.algorithms` | GRPO, DPO, SFT, losses, advantage/reward processing |
| Task plugins | `nemo_rl.tasks` | Dataset builders, task setup, task specs |
| Compatibility envs | `nemo_rl.environments` | Ray environments and legacy import paths |
| Recipes | `examples/configs` | Composable experiment configs and legacy configs |

Key extension points:

- `EnvStepResult`: normalized environment output with rewards, termination flags,
  stop strings, and metrics.
- `RolloutResult`: structured rollout result for new code while legacy rollout
  functions still return `(batch, metrics)`.
- `PostTrainingTaskSpec`: framework-level description of a reusable RL task.
- `TaskSetupResult`: datasets and environment maps consumed by algorithms.

## Current Task Plugin: InfiniSST

The first task plugin is `nemo_rl.tasks.infinisst`.

It contains:

- `data.py`: InfiniSST datum and iterable dataset construction.
- `setup.py`: Ray actor and dataset setup.
- `task.py`: `INFINISST_TASK_SPEC`, the reusable framework task spec.

The old environment import remains available:

```python
from nemo_rl.environments.games.infinisst import InfiniSSTEnv
```

New framework code should prefer:

```python
from nemo_rl.tasks.infinisst import build_infinisst_task_spec

task = build_infinisst_task_spec()
setup_result = task.setup_data(
    tokenizer=tokenizer,
    env_cfg=config["env"],
    data_cfg=config["data"],
    task_name=task.name,
    length=train_length,
    val_length=val_length,
)
```

## Quick Start

Use the project container or an environment with the repo dependencies installed.
The local lightweight framework modules can import without `torch` or `ray`, but
training requires the full RL stack.

Install dependencies:

```bash
uv sync --extra infinisst
```

Run focused framework tests:

```bash
uv run pytest \
  tests/unit/posttraining \
  tests/unit/tasks/infinisst \
  tests/unit/algorithms/test_grpo_components.py
```

Run InfiniSST GRPO locally inside a prepared container:

```bash
uv run python examples/run_grpo_infinisst.py \
  --config examples/configs/grpo_infinisst_4b_modular.yaml
```

Run through the legacy Slurm launcher:

```bash
bash docker_sbatch_3node.sh grpo_infinisst_4b_modular
```

The modular recipe composes:

- `examples/configs/infinisst/base_grpo_4b.yaml`
- `examples/configs/infinisst/language/en_zh.yaml`
- `examples/configs/infinisst/scoring/metricx_24.yaml`

Legacy configs under `examples/configs/grpo_infinisst_*.yaml` remain supported.

## Adding A New Speech RL Task

Create a package under `nemo_rl/tasks/<task_name>/`:

```text
nemo_rl/tasks/<task_name>/
  __init__.py
  data.py
  setup.py
  task.py
```

Implement:

- `data.py`: build training/validation examples and task metadata.
- `setup.py`: create datasets and Ray environments.
- `task.py`: expose a `PostTrainingTaskSpec`.

Then add a config group:

```text
examples/configs/<task_name>/
  base.yaml
  reward/<reward_backend>.yaml
  language/<language_pair>.yaml
```

The task should be usable by GRPO without modifying
`nemo_rl.algorithms.grpo`.

## Compatibility Guarantees

The refactor keeps these existing entrypoints available:

- `examples/run_grpo_infinisst.py --config ...`
- `nemo_rl.algorithms.grpo.setup`
- `nemo_rl.algorithms.grpo.grpo_train`
- `nemo_rl.experience.rollouts.run_multi_turn_rollout`
- `nemo_rl.experience.rollouts.run_async_multi_turn_rollout`
- `nemo_rl.environments.games.infinisst.InfiniSSTEnv`

This lets old experiments continue to run while new code moves toward the task
plugin API.

## Development Notes

- Keep task dependencies lazy-loaded where possible so framework modules remain
  lightweight.
- Keep reusable task logic in `nemo_rl.tasks`, not in `examples/`.
- Keep algorithm changes in `nemo_rl.algorithms`, not inside task environments.
- Use modular configs for new experiments.
- Use `docs/design-docs/rl-posttraining-framework.md` as the design reference.

Recommended checks before pushing:

```bash
python3 -m py_compile \
  nemo_rl/posttraining/protocol.py \
  nemo_rl/posttraining/task.py \
  nemo_rl/tasks/infinisst/task.py \
  examples/run_grpo_infinisst.py

git diff --check
```
