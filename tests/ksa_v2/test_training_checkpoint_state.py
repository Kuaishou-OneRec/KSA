from types import SimpleNamespace

import pytest
import torch


def test_weights_only_checkpoint_does_not_advance_fresh_adam():
    from muse.training.checkpoint import AppState

    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters())
    state = AppState(model, optimizer, skip_optimizer=True).state_dict()
    assert 'optim' not in state
    assert not optimizer.state
    assert all(parameter.grad is None for parameter in model.parameters())

    model(torch.ones(2, 4)).square().sum().backward()
    optimizer.step()
    assert all(int(value['step']) == 1 for value in optimizer.state.values())


def test_app_state_includes_training_state_roundtrip(monkeypatch):
    from muse.training import checkpoint as checkpoint_mod

    model = SimpleNamespace()
    optimizer = SimpleNamespace()
    training_state = {
        "step_scheduler": {"micro_step": 8, "global_step": 2},
        "lr_scheduler": {"last_epoch": 2},
    }

    monkeypatch.setattr(
        checkpoint_mod,
        "get_state_dict",
        lambda model, optimizer: ({"model_weight": 1}, {"optim_state": 2}),
    )

    loaded = {}

    def fake_set_state_dict(model, optimizer, model_state_dict, optim_state_dict):
        loaded["model"] = model_state_dict
        loaded["optim"] = optim_state_dict

    monkeypatch.setattr(checkpoint_mod, "set_state_dict", fake_set_state_dict)

    app_state = checkpoint_mod.AppState(
        model=model,
        optimizer=optimizer,
        training_state=training_state,
    )
    state = app_state.state_dict()

    assert state["training"] == training_state

    restored = checkpoint_mod.AppState(model=model, optimizer=optimizer, training_state={})
    restored.load_state_dict(state)

    assert restored.training_state == training_state
    assert loaded["model"] == {"model_weight": 1}
    assert loaded["optim"] == {"optim_state": 2}


def test_app_state_serializes_and_restores_stateful_training_components(monkeypatch):
    from muse.training import checkpoint as checkpoint_mod

    class DummyStateful:
        def __init__(self, value):
            self.value = value

        def state_dict(self):
            return {"value": self.value}

        def load_state_dict(self, state_dict):
            self.value = state_dict["value"]

    monkeypatch.setattr(
        checkpoint_mod,
        "get_state_dict",
        lambda model, optimizer: ({"model_weight": 1}, {"optim_state": 2}),
    )
    monkeypatch.setattr(
        checkpoint_mod,
        "set_state_dict",
        lambda model, optimizer, model_state_dict, optim_state_dict: None,
    )

    scheduler = DummyStateful(7)
    lr_scheduler = DummyStateful(3)
    app_state = checkpoint_mod.AppState(
        model=SimpleNamespace(),
        optimizer=SimpleNamespace(),
        training_state={
            "step_scheduler": scheduler,
            "lr_scheduler": lr_scheduler,
        },
    )

    state = app_state.state_dict()
    assert state["training"]["step_scheduler"] == {"value": 7}
    assert state["training"]["lr_scheduler"] == {"value": 3}

    restored_scheduler = DummyStateful(0)
    restored_lr_scheduler = DummyStateful(0)
    restored = checkpoint_mod.AppState(
        model=SimpleNamespace(),
        optimizer=SimpleNamespace(),
        training_state={
            "step_scheduler": restored_scheduler,
            "lr_scheduler": restored_lr_scheduler,
        },
    )
    restored.load_state_dict(state)

    assert restored_scheduler.value == 7
    assert restored_lr_scheduler.value == 3


def test_app_state_skip_optimizer_still_restores_training_state(monkeypatch):
    """skip_optimizer=True skips optimizer but still restores training state."""
    from muse.training import checkpoint as checkpoint_mod

    class DummyStateful:
        def __init__(self, value):
            self.value = value

        def state_dict(self):
            return {"value": self.value}

        def load_state_dict(self, state_dict):
            self.value = state_dict["value"]

    calls = {}

    def fake_set_state_dict(model, optimizer, model_state_dict, optim_state_dict):
        calls["model"] = model_state_dict
        calls["optim"] = optim_state_dict

    monkeypatch.setattr(checkpoint_mod, "set_state_dict", fake_set_state_dict)

    restored_scheduler = DummyStateful(0)
    restored_lr_scheduler = DummyStateful(0)
    restored = checkpoint_mod.AppState(
        model=SimpleNamespace(),
        optimizer=SimpleNamespace(),
        training_state={
            "step_scheduler": restored_scheduler,
            "lr_scheduler": restored_lr_scheduler,
        },
        skip_optimizer=True,
    )

    restored.load_state_dict({
        "model": {"model_weight": 1},
        "optim": {"optim_state": 2},
        "training": {
            "step_scheduler": {"value": 7},
            "lr_scheduler": {"value": 3},
        },
    })

    # Optimizer is skipped
    assert calls["model"] == {"model_weight": 1}
    assert calls["optim"] == {}
    # Training state IS restored (decoupled from optimizer skip)
    assert restored_scheduler.value == 7
    assert restored_lr_scheduler.value == 3


def test_app_state_skip_optimizer_with_skip_training_state_keys(monkeypatch):
    """skip_optimizer + skip_training_state_keys reproduces old load_weights_only behavior."""
    from muse.training import checkpoint as checkpoint_mod

    class DummyStateful:
        def __init__(self, value):
            self.value = value

        def state_dict(self):
            return {"value": self.value}

        def load_state_dict(self, state_dict):
            self.value = state_dict["value"]

    calls = {}

    def fake_set_state_dict(model, optimizer, model_state_dict, optim_state_dict):
        calls["model"] = model_state_dict
        calls["optim"] = optim_state_dict

    monkeypatch.setattr(checkpoint_mod, "set_state_dict", fake_set_state_dict)

    restored_scheduler = DummyStateful(0)
    restored_lr_scheduler = DummyStateful(0)
    restored = checkpoint_mod.AppState(
        model=SimpleNamespace(),
        optimizer=SimpleNamespace(),
        training_state={
            "step_scheduler": restored_scheduler,
            "lr_scheduler": restored_lr_scheduler,
        },
        skip_optimizer=True,
        skip_training_state_keys=["step_scheduler", "lr_scheduler"],
    )

    restored.load_state_dict({
        "model": {"model_weight": 1},
        "optim": {"optim_state": 2},
        "training": {
            "step_scheduler": {"value": 7},
            "lr_scheduler": {"value": 3},
        },
    })

    # Both optimizer and training state are skipped
    assert calls["model"] == {"model_weight": 1}
    assert calls["optim"] == {}
    assert restored_scheduler.value == 0
    assert restored_lr_scheduler.value == 0


def test_app_state_can_skip_selected_training_state_keys_on_load(monkeypatch):
    from muse.training import checkpoint as checkpoint_mod

    class DummyStateful:
        def __init__(self, value):
            self.value = value

        def state_dict(self):
            return {"value": self.value}

        def load_state_dict(self, state_dict):
            self.value = state_dict["value"]

    monkeypatch.setattr(
        checkpoint_mod,
        "set_state_dict",
        lambda model, optimizer, model_state_dict, optim_state_dict: None,
    )

    restored_scheduler = DummyStateful(0)
    restored_dataloader = DummyStateful(100)
    restored = checkpoint_mod.AppState(
        model=SimpleNamespace(),
        optimizer=SimpleNamespace(),
        training_state={
            "step_scheduler": restored_scheduler,
            "dataloader": restored_dataloader,
        },
        skip_training_state_keys=["dataloader"],
    )

    restored.load_state_dict({
        "model": {"model_weight": 1},
        "optim": {"optim_state": 2},
        "training": {
            "step_scheduler": {"value": 7},
            "dataloader": {"value": 3},
        },
    })

    assert restored_scheduler.value == 7
    assert restored_dataloader.value == 100


def test_app_state_requires_expected_training_state_keys(monkeypatch):
    from muse.training import checkpoint as checkpoint_mod

    monkeypatch.setattr(
        checkpoint_mod,
        "set_state_dict",
        lambda model, optimizer, model_state_dict, optim_state_dict: None,
    )

    restored = checkpoint_mod.AppState(
        model=SimpleNamespace(),
        optimizer=SimpleNamespace(),
        training_state={"dataloader": SimpleNamespace()},
        required_training_state_keys=["dataloader"],
    )

    with pytest.raises(KeyError, match="dataloader"):
        restored.load_state_dict({
            "model": {"model_weight": 1},
            "optim": {"optim_state": 2},
            "training": {},
        })


class _DictStateful:
    def __init__(self, model_state):
        self.model_state = model_state

    def state_dict(self):
        return {"model": self.model_state}

    def load_state_dict(self, state_dict):
        self.model_state = state_dict["model"]


def test_dcp_load_keeps_explicitly_allowed_missing_model_params(tmp_path):
    import torch.distributed.checkpoint as dcp

    from muse.training.checkpoint import DistributedCheckpointer

    checkpoint_dir = tmp_path / "checkpoint"
    source = _DictStateful({"model.base.weight": torch.tensor([3.0])})
    dcp.save({"app": source}, checkpoint_id=checkpoint_dir)

    target = _DictStateful({
        "model.base.weight": torch.tensor([0.0]),
        "model.layers.0.attn.meta_pool_scorer.weight": torch.tensor([7.0]),
    })
    DistributedCheckpointer().load_checkpoint(
        {"app": target},
        str(checkpoint_dir),
        allow_random_init_params="meta_pool_scorer.weight",
    )

    assert target.model_state["model.base.weight"].item() == 3.0
    assert (
        target.model_state[
            "model.layers.0.attn.meta_pool_scorer.weight"
        ].item()
        == 7.0
    )


def test_dcp_load_rejects_unapproved_missing_model_params(tmp_path):
    import torch.distributed.checkpoint as dcp

    from muse.training.checkpoint import DistributedCheckpointer

    checkpoint_dir = tmp_path / "checkpoint"
    source = _DictStateful({"model.base.weight": torch.tensor([3.0])})
    dcp.save({"app": source}, checkpoint_id=checkpoint_dir)

    target = _DictStateful({
        "model.base.weight": torch.tensor([0.0]),
        "model.unexpected_new.weight": torch.tensor([7.0]),
    })
    with pytest.raises(RuntimeError, match="unexpected_new.weight"):
        DistributedCheckpointer().load_checkpoint(
            {"app": target},
            str(checkpoint_dir),
            allow_random_init_params="meta_pool_scorer.weight",
        )
