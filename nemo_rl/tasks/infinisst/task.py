# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

from nemo_rl.posttraining.task import PostTrainingTaskSpec, TaskSetupResult
from nemo_rl.tasks.infinisst.setup import setup_infinisst_data

INFINISST_TASK_NAME = "infinisst"


def setup_infinisst_task(
    tokenizer: Any,
    env_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    task_name: str = INFINISST_TASK_NAME,
    length: int = 0,
    val_length: int = 0,
) -> TaskSetupResult:
    train_dataset, val_dataset, task_to_env, val_task_to_env = setup_infinisst_data(
        tokenizer=tokenizer,
        env_cfg=env_cfg,
        data_cfg=data_cfg,
        task_name=task_name,
        length=length,
        val_length=val_length,
    )
    return TaskSetupResult(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task_to_env=task_to_env,
        val_task_to_env=val_task_to_env,
    )


def build_infinisst_task_spec() -> PostTrainingTaskSpec:
    return PostTrainingTaskSpec(
        name=INFINISST_TASK_NAME,
        setup_data=setup_infinisst_task,
        description=(
            "Streaming speech translation RL task built on the HPO/InfiniSST "
            "environment, exposed as a reusable posttraining framework plugin."
        ),
        default_config_paths=(
            "examples/configs/grpo_infinisst_4b_modular.yaml",
        ),
        reward_metrics=(
            "quality",
            "latency",
            "hinged_latency",
        ),
        tags=(
            "speech",
            "simultaneous-translation",
            "grpo",
        ),
    )


INFINISST_TASK_SPEC = build_infinisst_task_spec()
