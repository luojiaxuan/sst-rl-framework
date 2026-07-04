import torch

from nemo_rl.algorithms.grpo_components import compute_grpo_advantages


def test_compute_grpo_advantages_can_use_raw_rewards():
    output = compute_grpo_advantages(
        input_features=[["a"], ["a"]],
        rewards=torch.tensor([1.0, 3.0]),
        use_leave_one_out_baseline=False,
        reduce_baseline_rewards=False,
        normalize_rewards=False,
    )

    assert torch.allclose(output.advantages, torch.tensor([[1.0], [3.0]]))


def test_compute_grpo_advantages_reduces_prompt_baseline():
    output = compute_grpo_advantages(
        input_features=[["a"], ["a"]],
        rewards=torch.tensor([1.0, 3.0]),
        use_leave_one_out_baseline=False,
        reduce_baseline_rewards=True,
        normalize_rewards=False,
    )

    assert torch.allclose(output.baseline, torch.tensor([2.0, 2.0]))
    assert torch.allclose(output.advantages, torch.tensor([[-1.0], [1.0]]))


def test_compute_grpo_advantages_normalizes_nonzero_std():
    output = compute_grpo_advantages(
        input_features=[["a"], ["a"]],
        rewards=torch.tensor([1.0, 3.0]),
        use_leave_one_out_baseline=False,
        reduce_baseline_rewards=True,
        normalize_rewards=True,
    )

    assert torch.allclose(output.std, torch.tensor([1.0, 1.0]))
    assert torch.allclose(output.advantages, torch.tensor([[-1.0], [1.0]]))
