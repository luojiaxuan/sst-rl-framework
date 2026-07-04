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

from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.tasks.infinisst.data import IterableInfiniSSTDataset


def _build_infinisst_actor_runtime_env(env_config: dict[str, Any]) -> dict[str, Any]:
    runtime_env = dict(env_config.get("actor_runtime_env", {}))
    runtime_env.setdefault(
        "py_executable",
        get_actor_python_env("nemo_rl.environments.games.infinisst.InfiniSSTEnv"),
    )
    return runtime_env


def setup_infinisst_data(
    tokenizer: PreTrainedTokenizerBase,
    env_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    task_name: str,
    length: int,
    val_length: int,
) -> tuple[IterableDataset, IterableDataset | None, dict, dict]:
    from nemo_rl.environments.games.infinisst import InfiniSSTEnv

    print("Setting up InfiniSST data and environment.")
    env_config = env_cfg[task_name]
    env = InfiniSSTEnv.options(
        runtime_env=_build_infinisst_actor_runtime_env(env_config)
    ).remote(cfg=dict(env_config["cfg"]))
    task_to_env = {task_name: env}

    training_dataset = IterableInfiniSSTDataset(
        tokenizer=tokenizer,
        data_file=data_cfg["train_data_file"],
        shuffle=data_cfg["train_data_shuffle"],
        seed=data_cfg["seed"],
        src_lang=data_cfg["src_lang"],
        tgt_lang=data_cfg["tgt_lang"],
        task_name=task_name,
        length=length,
        multiplier=data_cfg.get("multiplier", 1),
    )

    validation_dataset = IterableInfiniSSTDataset(
        tokenizer=tokenizer,
        data_file=data_cfg["val_data_file"],
        shuffle=data_cfg["val_data_shuffle"],
        seed=data_cfg["seed"],
        src_lang=data_cfg["src_lang"],
        tgt_lang=data_cfg["tgt_lang"],
        task_name=task_name,
        length=val_length,
        multiplier=data_cfg.get("multiplier", 1),
    )
    return training_dataset, validation_dataset, task_to_env, task_to_env
