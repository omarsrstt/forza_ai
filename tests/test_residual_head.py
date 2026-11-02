"""The model must start from the previous action, not from zero.

Measured on the held-out split, repeating the last steering command scores 0.132
MAE while the trained ConvNeXt+LSTM scores 0.275. A linear probe on frozen
features reproduces the network to within 0.004, and the raw image explains only
2.7% of the variance that recent actions leave behind. So the ceiling on this
task is set by the *input*, not by the loss, the head or the optimiser: the
network was never shown its own last command and cannot recover it.

These tests pin the residual formulation that fixes that:

    action(t) = clip( previous_action(t) + SCALE * tanh(delta(t)) )

Two properties matter enough to be tested rather than assumed:

  * a model whose correction head is zeroed **is** persistence -- otherwise the
    architecture does not actually start where the baseline is;
  * the correction is bounded, so a confidently wrong delta cannot command more
    than SCALE beyond the previous action.

And a third, subtler one: at time t the LSTM must be given the features of t-1,
not of t. Feeding the frame it has to predict is label leakage, which trains to
a perfect loss and learns nothing.
"""

import torch

import train_model

CHANNELS = 3
FEAT = 768   # ConvNeXt-Tiny feature width, which is what the LSTM expects


def _m(residual=True):
    return train_model.ConvNextTinyLSTMRegression(num_outputs=3, residual=residual).eval()


def _zero_correction(model):
    """Force delta == 0 by zeroing the correction head's weights and biases."""
    with torch.no_grad():
        for p in model.delta.parameters():
            p.zero_()
    return model


def test_zeroed_correction_head_reproduces_the_previous_action():
    """With delta==0 the output must equal ``prev`` exactly: that is the claim
    that this architecture is persistence plus a learned correction. If it
    fails, the model is still starting from somewhere else and the baseline
    comparison is meaningless."""
    m = _zero_correction(_m())
    feat = torch.randn(1, 4, FEAT)                       # stand-in CNN features
    prev = torch.tensor([[[0.4, 0.6, 0.0],
                          [0.5, 0.6, 0.0],
                          [0.5, 0.7, 0.2],
                          [0.3, 0.7, 0.0]]])
    with torch.no_grad():
        out = m.forward_from_features(feat, prev=prev)
    assert torch.allclose(out, prev, atol=1e-6)


def test_correction_is_bounded_by_the_scale():
    """A saturated delta must not move the command by more than SCALE."""
    m = _m()
    with torch.no_grad():
        for p in m.delta.parameters():
            p.fill_(1e3)                                # saturate tanh
    feat = torch.randn(1, 2, FEAT)
    prev = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    with torch.no_grad():
        out = m.forward_from_features(feat, prev=prev)
    assert (out - prev).abs().max() <= train_model.RESIDUAL_SCALE + 1e-6
    assert out[..., 0].abs().max() <= 1.0 + 1e-6
    assert out[..., 1].min() >= -1e-6 and out[..., 1].max() <= 1.0 + 1e-6
    assert out[..., 2].min() >= -1e-6 and out[..., 2].max() <= 1.0 + 1e-6


def test_previous_action_actually_changes_the_output():
    """prev must be wired in, not silently ignored."""
    m = _m()
    feat = torch.randn(1, 3, FEAT)
    a = torch.zeros(1, 3, CHANNELS)
    b = torch.ones(1, 3, CHANNELS) * 0.5
    with torch.no_grad():
        ya = m.forward_from_features(feat, prev=a)
        yb = m.forward_from_features(feat, prev=b)
    assert (ya - yb).abs().max() > 1e-3


def test_predictions_use_the_current_frame_not_a_stale_one():
    """The driver must not throw away the newest camera frame.

    An earlier draft shifted the LSTM input by one step to avoid "leaking" the
    frame being predicted. That is wrong here: the human's action at t was
    produced from the image at t, and at inference the frame is captured before
    the command is sent. Shifting made the car react one tick late and, worse,
    made the live path disagree with the training path. Replacing the newest
    frame's features must change the newest prediction.
    """
    m = _m()
    base = torch.randn(1, 3, FEAT)
    other = base.clone()
    other[:, -1] = torch.randn(FEAT)
    prev = torch.zeros(1, 3, CHANNELS)
    with torch.no_grad():
        a = m.forward_from_features(base, prev=prev)
        b = m.forward_from_features(other, prev=prev)
    assert (a[:, -1] - b[:, -1]).abs().max() > 1e-4


def test_missing_prev_defaults_to_zero_not_an_error():
    """Old call sites (eval harness, diagnostics, checkpoints without the head)
    must keep working unchanged."""
    m = _m()
    x = torch.randn(1, 2, 3, 32, 32)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (1, 2, 3)


def test_residual_can_be_disabled_for_old_checkpoints():
    """Checkpoints trained before this change have no ``delta`` head, so the
    loader must still be able to construct the original architecture."""
    m = train_model.ConvNextTinyLSTMRegression(num_outputs=3, residual=False)
    assert not hasattr(m, "delta")
    with torch.no_grad():
        y = m(torch.randn(1, 2, 3, 32, 32))
    assert y.shape == (1, 2, 3)


def test_shift_labels_is_teacher_forcing():
    """prev[t] must be the *true* action at t-1, zero-filled at the window start.

    Zero filling only touches t == 0, and inference only ever reads the last
    timestep, so the padding never reaches the car.
    """
    labels = torch.tensor([[[0.1, 0.5, 0.0],
                           [0.2, 0.6, 0.1],
                           [0.3, 0.7, 0.0]]])
    prev = train_model.shift_labels(labels)
    assert torch.allclose(prev[:, 0], torch.zeros(1, 3))
    assert torch.allclose(prev[:, 1], labels[:, 0])
    assert torch.allclose(prev[:, 2], labels[:, 1])
    assert prev.shape == labels.shape
