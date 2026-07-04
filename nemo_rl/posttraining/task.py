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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class TaskSetupResult:
    """Datasets and environment maps required by a posttraining algorithm."""

    train_dataset: Any
    val_dataset: Any | None
    task_to_env: dict[str, Any]
    val_task_to_env: dict[str, Any] | None = None


TaskSetupFn = Callable[..., TaskSetupResult]


@dataclass(frozen=True)
class PostTrainingTaskSpec:
    """Framework-level description of a reusable RL posttraining task."""

    name: str
    setup_data: TaskSetupFn
    description: str = ""
    default_config_paths: tuple[str, ...] = ()
    reward_metrics: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass
class TaskRegistry:
    """Small in-process registry for task plugins."""

    _tasks: dict[str, PostTrainingTaskSpec] = field(default_factory=dict)

    def register(self, task: PostTrainingTaskSpec) -> PostTrainingTaskSpec:
        if task.name in self._tasks:
            raise ValueError(f"Task already registered: {task.name}")
        self._tasks[task.name] = task
        return task

    def get(self, name: str) -> PostTrainingTaskSpec:
        try:
            return self._tasks[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._tasks)) or "<empty>"
            raise KeyError(f"Unknown posttraining task {name!r}. Known tasks: {known}") from exc

    def names(self) -> list[str]:
        return sorted(self._tasks)


DEFAULT_TASK_REGISTRY = TaskRegistry()


def register_task(task: PostTrainingTaskSpec) -> PostTrainingTaskSpec:
    return DEFAULT_TASK_REGISTRY.register(task)


def get_task(name: str) -> PostTrainingTaskSpec:
    return DEFAULT_TASK_REGISTRY.get(name)
