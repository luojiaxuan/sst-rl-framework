from nemo_rl.tasks.infinisst.task import (
    INFINISST_TASK_NAME,
    build_infinisst_task_spec,
)


def test_infinisst_task_spec_exposes_framework_metadata():
    task = build_infinisst_task_spec()

    assert task.name == INFINISST_TASK_NAME
    assert "speech" in task.tags
    assert "grpo" in task.tags
    assert task.default_config_paths == (
        "examples/configs/grpo_infinisst_4b_modular.yaml",
    )
    assert "latency" in task.reward_metrics
