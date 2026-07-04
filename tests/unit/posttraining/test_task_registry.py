import pytest

from nemo_rl.posttraining.task import (
    PostTrainingTaskSpec,
    TaskRegistry,
    TaskSetupResult,
)


def _setup_stub(**kwargs):
    return TaskSetupResult(
        train_dataset="train",
        val_dataset="val",
        task_to_env={"stub": "env"},
        val_task_to_env={"stub": "val_env"},
    )


def test_task_registry_registers_and_lists_tasks():
    registry = TaskRegistry()
    task = PostTrainingTaskSpec(
        name="stub",
        setup_data=_setup_stub,
        description="test task",
        tags=("unit",),
    )

    registry.register(task)

    assert registry.names() == ["stub"]
    assert registry.get("stub") is task


def test_task_registry_rejects_duplicate_names():
    registry = TaskRegistry()
    task = PostTrainingTaskSpec(name="stub", setup_data=_setup_stub)

    registry.register(task)

    with pytest.raises(ValueError, match="Task already registered"):
        registry.register(task)


def test_task_spec_setup_returns_framework_result():
    task = PostTrainingTaskSpec(name="stub", setup_data=_setup_stub)

    result = task.setup_data()

    assert result.train_dataset == "train"
    assert result.task_to_env == {"stub": "env"}
