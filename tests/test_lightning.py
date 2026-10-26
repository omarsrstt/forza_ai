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
