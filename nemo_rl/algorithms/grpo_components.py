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

from typing import Any, NamedTuple

import torch

from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt


class GRPOAdvantageOutput(NamedTuple):
    advantages: torch.Tensor
    baseline: torch.Tensor
    std: torch.Tensor


def compute_grpo_advantages(
    input_features: list[Any],
    rewards: torch.Tensor,
    use_leave_one_out_baseline: bool,
    reduce_baseline_rewards: bool,
    normalize_rewards: bool,
) -> GRPOAdvantageOutput:
    baseline, std = calculate_baseline_and_std_per_prompt(
        input_features,
        rewards,
        torch.ones_like(rewards),
        leave_one_out_baseline=use_leave_one_out_baseline,
    )

    if reduce_baseline_rewards:
        advantages = (rewards - baseline).unsqueeze(-1)
    else:
        advantages = rewards.unsqueeze(-1)

    if normalize_rewards:
        zero_std_mask = std > 0
        advantages[zero_std_mask] = (
            advantages[zero_std_mask] / std.unsqueeze(-1)[zero_std_mask]
        )

    return GRPOAdvantageOutput(
        advantages=advantages,
        baseline=baseline,
        std=std,
    )
