"""Closed-loop scoring: feed the model its own outputs, as the driver does.

Every existing metric is teacher-forced: ``prev`` is the human's true previous
action. The car never has that -- drive.py feeds back the command it actually
sent. If the correction head has any gain, self-feeding errors compound and the
teacher-forced number is optimistic by an unknown margin. That difference is
invisible to every metric used so far.

The rollout is exact rather than approximate because the correction depends only
on the causal LSTM output, never on ``prev``: the recurrence is
``out[t] = clip(out[t-1] + delta[t])`` over a single LSTM pass.
"""

import torch

import train_model

CH = 3
FEAT = 768


def _m():
    return train_model.ConvNextTinyLSTMRegression(num_outputs=3, residual=True).eval()


def _zero_head(m):
    with torch.no_grad():
        for p in m.delta.parameters():
            p.zero_()
    return m


def test_closed_loop_with_zero_correction_holds_the_initial_action():
    """No correction means "keep doing what you were doing" forever -- this is the
    persistence policy, and it is the floor the rollout must reproduce."""
    m = _zero_head(_m())
    feat = torch.randn(1, 5, FEAT)
    init = torch.tensor([[[0.4, 0.9, 0.0]]])
    with torch.no_grad():
        out = m.forward_closed_loop(feat, init_prev=init)
    assert out.shape == (1, 5, CH)
    for t in range(5):
        assert torch.allclose(out[:, t], init[:, 0], atol=1e-6)


def test_closed_loop_first_step_matches_the_open_loop_prediction():
    """At t=0 there is no self-feeding yet, so both formulations must agree."""
    m = _m()
    feat = torch.randn(1, 3, FEAT)
    init = torch.tensor([[[0.2, 0.7, 0.0]]])
    with torch.no_grad():
        closed = m.forward_closed_loop(feat, init_prev=init)
        opened = m.forward_from_features(feat, prev=init)
    assert torch.allclose(closed[:, 0], opened[:, 0], atol=1e-5)


def test_closed_loop_actually_feeds_itself_forward():
    """The rollout must accumulate its own corrections, not repeat one step.

    With a non-zero head, step 2 depends on step 1's output; if the implementation
    silently kept using ``init_prev`` the two would be identical.
    """
    m = _m()
    feat = torch.randn(1, 4, FEAT)
    init = torch.zeros(1, 1, CH)
    with torch.no_grad():
        out = m.forward_closed_loop(feat, init_prev=init)
        delta = (out - torch.cat([init, out[:, :-1]], dim=1)).abs().max()
    assert delta > 1e-6
    # and it must stay in range
    assert out[..., 0].abs().max() <= 1.0 + 1e-6
    assert out[..., 1].min() >= -1e-6 and out[..., 1].max() <= 1.0 + 1e-6


def test_closed_loop_defaults_to_standstill():
    """The driver starts from a stationary, centred car."""
    m = _zero_head(_m())
    with torch.no_grad():
        out = m.forward_closed_loop(torch.randn(1, 3, FEAT))
    assert torch.allclose(out, torch.zeros(1, 3, CH), atol=1e-6)


def test_closed_loop_matches_teacher_forced_when_the_model_is_right():
    """If the model's predictions equal the labels, self-feeding changes nothing:
    the gap between the two metrics is purely compounding error."""
    m = _m()
    feat = torch.randn(1, 4, FEAT)
    with torch.no_grad():
        target = m.forward_closed_loop(feat)
        forced = m.forward_from_features(feat, prev=torch.cat(
            [torch.zeros(1, 1, CH), target[:, :-1]], dim=1))
    assert torch.allclose(forced, target, atol=1e-5)
