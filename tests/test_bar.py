"""The acceptance bar must include the persistence baseline.

The v1 model passed every existing check -- beat constant-mean MAE, responded to
its input, even cleared the sensitivity gate -- while being 2x worse than simply
repeating the last command. Every check it was held to was weaker than a one-line
baseline, so "PASS" meant nothing.

Persistence is not a theoretical baseline here: the driver knows the last command
it sent, so `y[t] = y[t-1]` is a predictor that genuinely exists at inference.
These tests make it impossible to certify a model that loses to it.
"""

import torch

import eval_model


class _Ramp:
    """Labels ramp by a fixed step per frame, so persistence has a known error."""

    def __init__(self, n=4, k=5, step=0.2):
        self.n, self.k, self.step = n, k, step

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        steer = torch.arange(self.k, dtype=torch.float32) * self.step
        labels = torch.stack(
            [steer, torch.full((self.k,), 0.5), torch.zeros(self.k)], dim=1
        )
        return torch.zeros(self.k, 3, 4, 4), labels


class _CopyPrev:
    """Predicts the previous label exactly: the ideal persistence model."""

    def forward(self, x):
        return None


def test_persistence_baseline_is_measured_on_real_labels():
    """On a constant-step ramp, predicting t-1 is off by exactly one step."""
    mae = eval_model.persistence_baseline_mae(_Ramp(step=0.2))
    assert mae["steer"] == torch.tensor(0.2).item() or abs(mae["steer"] - 0.2) < 1e-5


def test_persistence_mae_is_zero_for_a_constant_sequence():
    """Nothing changes, so repeating the last value is exact."""
    mae = eval_model.persistence_baseline_mae(_Ramp(step=0.0))
    assert mae["steer"] < 1e-6


def test_bar_fails_a_model_that_loses_to_persistence():
    """This is the exact failure mode that shipped: good MAE versus the constant
    baseline, but worse than holding the last command."""
    passed, checks = eval_model.meets_bar(
        {"steer": 0.27, "throttle": 0.10, "brake": 0.04},     # the v1 numbers
        {"steer": 0.41},                                       # beats constant mean
        {"steer": 1.0, "throttle": 1.0},
        {"steer": (0.95, 0.90)},                               # calibration clears
        persistence={"steer": 0.13},                           # but persistence wins
    )
    assert checks["steer_beats_persistence"] is False
    assert passed is False


def test_bar_passes_a_model_that_beats_persistence():
    passed, checks = eval_model.meets_bar(
        {"steer": 0.10, "throttle": 0.05, "brake": 0.02},
        {"steer": 0.41},
        {"steer": 1.0, "throttle": 1.0},
        {"steer": (0.95, 0.90)},
        persistence={"steer": 0.13},
    )
    assert checks["steer_beats_persistence"] is True
    assert passed is True


def test_bar_without_persistence_still_works():
    """Callers that do not compute the baseline keep the previous behaviour."""
    passed, checks = eval_model.meets_bar(
        {"steer": 0.3, "throttle": 0.5, "brake": 0.1},
        {"steer": 0.5},
        {"steer": 0.9, "throttle": 0.9},
        {"steer": (0.95, 0.90)},
    )
    assert "steer_beats_persistence" not in checks
    assert passed is True
