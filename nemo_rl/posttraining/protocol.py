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
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol, TypedDict

if TYPE_CHECKING:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _torch_module():
    import torch

    return torch


class Message(TypedDict, total=False):
    """OpenAI-style turn with optional tensors used by posttraining."""

    role: str
    content: str
    token_ids: torch.Tensor
    features: Any
    generation_logprobs: torch.Tensor
    advantages: torch.Tensor


@dataclass
class EpisodeState:
    """Single episode state passed between rollout and task environments."""

    message_log: list[Message]
    metadata: dict[str, Any] | None = None
    task_name: str = "default"
    stop_strings: list[str] | None = None
    idx: int | None = None
    loss_multiplier: float = 1.0


@dataclass
class EnvStepBatch:
    """Batched environment request for one policy-generated turn."""

    message_logs: list[list[Message]]
    metadata: list[dict[str, Any] | None]
    task_names: list[str] = field(default_factory=list)


@dataclass
class EnvStepResult:
    """Batched environment response with metrics normalized to dict lists."""

    observations: list[Message]
    metadata: list[dict[str, Any] | None]
    next_stop_strings: list[list[str] | None]
    rewards: torch.Tensor
    terminateds: torch.Tensor
    metrics: dict[str, list[float | int]] = field(default_factory=dict)

    def __iter__(self) -> Iterable[Any]:
        yield self.observations
        yield self.metadata
        yield self.next_stop_strings
        yield self.rewards
        yield self.terminateds
        yield self.metrics


@dataclass
class RolloutResult:
    """Structured rollout result returned by new runner implementations."""

    batch: BatchedDataDict
    metrics: dict[str, Any]

    def as_tuple(self) -> tuple[BatchedDataDict, dict[str, Any]]:
        return self.batch, self.metrics


class TaskEnvironment(Protocol):
    def step(self, batch: EnvStepBatch) -> EnvStepResult:
        raise NotImplementedError

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict[str, Any]]:
        raise NotImplementedError


def _as_tensor(value: Any, dtype: Any) -> Any:
    torch = _torch_module()
    if torch.is_tensor(value):
        return value.detach().cpu().to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def _metric_scalar(value: Any, index: int) -> float | int:
    torch = _torch_module()
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _normalize_metrics(metrics: Any, batch_size: int) -> dict[str, list[float | int]]:
    torch = _torch_module()
    if metrics is None:
        return {}
    if isinstance(metrics, Mapping):
        normalized: dict[str, list[float | int]] = {}
        for name, values in metrics.items():
            if torch.is_tensor(values):
                values = values.detach().cpu().tolist()
            if isinstance(values, (list, tuple)):
                normalized[str(name)] = list(values)
            else:
                normalized[str(name)] = [values for _ in range(batch_size)]
        return normalized
    if isinstance(metrics, list):
        normalized = {}
        for item in metrics:
            if item is None:
                continue
            for name, values in item.items():
                normalized.setdefault(str(name), [])
                if torch.is_tensor(values):
                    values = values.detach().cpu().tolist()
                if isinstance(values, (list, tuple)):
                    normalized[str(name)].extend(values)
                else:
                    normalized[str(name)].append(values)
        return normalized
    raise TypeError(f"Unsupported environment metrics type: {type(metrics)!r}")


def _fields_from_result(result: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    if isinstance(result, EnvStepResult):
        return (
            result.observations,
            result.metadata,
            result.next_stop_strings,
            result.rewards,
            result.terminateds,
            result.metrics,
        )
    if all(
        hasattr(result, name)
        for name in ("observations", "metadata", "rewards", "terminateds")
    ):
        return (
            result.observations,
            result.metadata,
            result.next_stop_strings,
            result.rewards,
            result.terminateds,
            getattr(result, "metrics", None),
        )
    if isinstance(result, (tuple, list)) and len(result) == 5:
        observations, metadata, next_stop_strings, rewards, terminateds = result
        return observations, metadata, next_stop_strings, rewards, terminateds, None
    if isinstance(result, (tuple, list)) and len(result) == 6:
        observations, metadata, next_stop_strings, rewards, terminateds, metrics = result
        return observations, metadata, next_stop_strings, rewards, terminateds, metrics
    raise TypeError(f"Unsupported environment step result type: {type(result)!r}")


def normalize_env_step_result(
    result: Any, batch_size: int | None = None
) -> EnvStepResult:
    torch = _torch_module()
    observations, metadata, next_stop_strings, rewards, terminateds, metrics = (
        _fields_from_result(result)
    )
    if batch_size is None:
        batch_size = len(observations)
    if next_stop_strings is None:
        next_stop_strings = [None] * batch_size
    return EnvStepResult(
        observations=list(observations),
        metadata=list(metadata),
        next_stop_strings=list(next_stop_strings),
        rewards=_as_tensor(rewards, torch.float32),
        terminateds=_as_tensor(terminateds, torch.bool),
        metrics=_normalize_metrics(metrics, batch_size),
    )
