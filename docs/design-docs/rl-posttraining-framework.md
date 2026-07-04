# RL Posttraining Framework

This repository keeps the HPO-style simultaneous translation experiments
compatible while making the RL infrastructure reusable for future speech tasks.

## Framework Boundary

The framework owns shared RL mechanics:

- policy and generation backends
- rollout collection
- environment stepping
- reward and metric normalization
- GRPO/DPO/SFT training loops
- checkpointing and logging

Task packages own domain-specific pieces:

- dataset construction
- task metadata
- Ray environment actor setup
- reward model dependencies
- task-specific configs

`nemo_rl.tasks.infinisst` is the first task plugin. New speech RL work should
follow that shape instead of editing GRPO internals.

## Public Extension Points

- `PostTrainingTaskSpec` describes a reusable task plugin.
- `TaskSetupResult` returns datasets and environment maps to algorithms.
- `EnvStepResult` normalizes environment output across old tuple returns and
  new typed responses.
- `RolloutResult` gives new runner code a structured return type while legacy
  rollout functions keep returning `(batch, metrics)`.

## Adding A Speech RL Task

1. Create `nemo_rl/tasks/<task_name>/`.
2. Implement dataset construction in `data.py`.
3. Implement environment setup in `setup.py`.
4. Expose a `PostTrainingTaskSpec` in `task.py`.
5. Keep heavy task dependencies inside setup or dataset constructors.
6. Add a modular config under `examples/configs/<task_name>/`.
7. Reuse `nemo_rl.algorithms.grpo` unless the algorithm itself changes.

The result should let a new task plug into GRPO without rewriting rollout
generation, reward collection, or policy training.
