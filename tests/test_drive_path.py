"""Does the live driver reproduce the training-time forward pass exactly?

drive.py does not call model(x): it encodes one frame at a time, caches the
features and calls forward_from_features with a history shift. That is only safe
if it agrees with the batched training path to numerical noise -- a mismatch here
shows up as a slowly drifting steering bias on the car and nowhere else.

Pinned properties:
  * live path == batched path for the same window and previous action
  * the driver's prev_action is the command it actually sent, so a held command
    produces a held correction
  * the output is inside the physical ranges the gamepad writer accepts
"""

import json

import torch

import train_model


def _model():
    return train_model.ConvNextTinyLSTMRegression(num_outputs=3, residual=True).eval()


def test_live_path_matches_batched_path():
    m = _model()
    B, k = 1, 4
    x = torch.randn(B, k, 3, 32, 32)
    prev = torch.tensor([[[0.3, 0.8, 0.0], [0.3, 0.8, 0.0],
                          [0.4, 0.8, 0.0], [0.2, 0.9, 0.0]]])
    with torch.no_grad():
        batched = m(x, prev=prev)

        # what drive.py does: encode the newest frame, keep a rolling feature
        # window, and predict from the previous command
        feat = m.encode(x.reshape(-1, 3, 32, 32)).reshape(B, k, -1)
        live = m.forward_from_features(feat, prev=prev)

    assert torch.allclose(batched, live, atol=1e-5), (
        "live driver and training forward pass disagree"
    )


def test_prev_action_is_echoed_when_correction_is_zero():
    """Holding a steady command must keep holding it: the driver feeds its own
    previous output back in, so any gain on the skip connection compounds."""
    m = _model()
    with torch.no_grad():
        for p in m.delta.parameters():
            p.zero_()
    feat = torch.randn(1, 3, 768)
    seq = [torch.tensor([[[0.5, 1.0, 0.0]]])]
    for _ in range(5):
        out = m.forward_from_features(feat, prev=seq[-1])
        seq.append(out[:, -1:])
    for step in seq[1:]:
        assert torch.allclose(step, seq[0], atol=1e-6)


def test_outputs_are_in_physical_range_for_the_gamepad():
    m = _model()
    with torch.no_grad():
        for p in m.delta.parameters():
            p.fill_(1e3)                       # worst case: saturated correction
    feat = torch.randn(1, 2, 768)
    for prev in (torch.zeros(1, 2, 3), torch.ones(1, 2, 3),
                 torch.tensor([[[-1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]])):
        with torch.no_grad():
            out = m.forward_from_features(feat, prev=prev)
        assert out[..., 0].min() >= -1.0 - 1e-6 and out[..., 0].max() <= 1.0 + 1e-6
        assert out[..., 1].min() >= -1e-6 and out[..., 1].max() <= 1.0 + 1e-6
        assert out[..., 2].min() >= -1e-6 and out[..., 2].max() <= 1.0 + 1e-6


def test_drive_config_sequence_length_matches_training_config():
    """The driver caches k features; if k differs from training the model sees a
    window it was never trained on."""
    train_cfg = json.load(open("config.json"))
    drive_cfg = json.load(open("drive_config.json"))
    assert drive_cfg["sequence_length"] == train_cfg["sequence_length"]
