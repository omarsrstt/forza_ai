import pytest
import torch
import train_model

def _batch(B=2, k=4):
    return torch.randn(B,k,3,64,64), torch.rand(B,k,3)

def test_loss_is_finite_scalar_and_dense():
    m = train_model.ForzaLightning(train_model.ConvNextTinyLSTMRegression(3))
    loss = m.training_step(_batch(), 0)
    assert loss.ndim == 0 and torch.isfinite(loss)

def test_val_metrics_logged():
    m = train_model.ForzaLightning(train_model.ConvNextTinyLSTMRegression(3))
    out = m.validation_step(_batch(), 0)
    assert {"steer","throttle","brake"} <= set(out["mae"].keys())

def test_optimizer_has_two_param_groups():
    m = train_model.ForzaLightning(train_model.ConvNextTinyLSTMRegression(3))
    opt = m.configure_optimizers()["optimizer"]
    assert len(opt.param_groups) == 2


def test_huber_delta_is_actually_applied():
    """The objective must be a genuine Huber, not a quadratic in disguise.

    Default ``F.huber_loss`` uses delta=1.0, and the targets live in [-1, 1], so
    the linear region never activates and the loss degenerates to plain MSE.
    With delta=0.2 an error of 0.5 costs 0.2*(0.5-0.1)=0.08, versus 0.125 at
    delta=1.0. Same error, different objective -- proves the parameter is wired.
    """
    pred = torch.full((1, 1, 1), 0.5)
    target = torch.zeros((1, 1, 1))
    tight = train_model.ForzaLightning(
        train_model.ConvNextTinyLSTMRegression(3), huber_delta=0.2
    )
    loose = train_model.ForzaLightning(
        train_model.ConvNextTinyLSTMRegression(3), huber_delta=1.0
    )
    assert tight._loss(pred, target).item() == pytest.approx(0.08, abs=1e-6)
    assert loose._loss(pred, target).item() == pytest.approx(0.125, abs=1e-6)


def test_default_huber_delta_is_not_degenerate():
    m = train_model.ForzaLightning(train_model.ConvNextTinyLSTMRegression(3))
    assert m.huber_delta == pytest.approx(train_model.DEFAULT_HUBER_DELTA)
    assert m.huber_delta < 1.0   # must reach the linear region inside [-1, 1]
