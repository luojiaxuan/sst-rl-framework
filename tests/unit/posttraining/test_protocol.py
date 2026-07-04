import pytest
import torch

from nemo_rl.posttraining.protocol import EnvStepResult, normalize_env_step_result


def test_normalize_env_step_result_accepts_current_result():
    result = EnvStepResult(
        observations=[{"role": "environment", "content": "ok"}],
        metadata=[{"step": 1}],
        next_stop_strings=[["</done>"]],
        rewards=torch.tensor([1.0]),
        terminateds=torch.tensor([True]),
        metrics={"quality": [0.5]},
    )

    normalized = normalize_env_step_result(result)

    assert normalized.observations == result.observations
    assert normalized.metadata == result.metadata
    assert torch.allclose(normalized.rewards, torch.tensor([1.0]))
    assert normalized.terminateds.dtype == torch.bool
    assert normalized.metrics == {"quality": [0.5]}


def test_normalize_env_step_result_accepts_legacy_five_tuple():
    normalized = normalize_env_step_result(
        (
            [{"role": "environment", "content": "ok"}],
            [{"step": 1}],
            None,
            [1.0],
            [True],
        )
    )

    assert normalized.next_stop_strings == [None]
    assert torch.allclose(normalized.rewards, torch.tensor([1.0]))
    assert normalized.metrics == {}


def test_normalize_env_step_result_accepts_six_tuple_with_metrics():
    normalized = normalize_env_step_result(
        (
            [{"role": "environment", "content": "ok"}],
            [{"step": 1}],
            [["stop"]],
            [1.0],
            [False],
            {"score": torch.tensor([2.0])},
        )
    )

    assert normalized.next_stop_strings == [["stop"]]
    assert normalized.metrics == {"score": [2.0]}


def test_normalize_env_step_result_rejects_unknown_shape():
    with pytest.raises(TypeError):
        normalize_env_step_result(("too", "short"))
