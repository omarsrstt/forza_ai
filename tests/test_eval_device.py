"""Evaluation must run on the GPU when one is available.

``per_channel_mae``, ``calibration_stats`` and ``input_sensitivity`` all default
to ``device="cpu"``. On the full val split that is not slow, it is unusable: a
full-val call took over 30 minutes and had to be killed, which is why earlier
comparisons were silently done on 200-window subsamples (and why those numbers
wobbled by +-0.05). Two consecutive full-val runs on CPU would have cost an hour
to answer a question that takes minutes on the GPU.

The device must therefore default to the best available one, while still
accepting an explicit override for tests and CPU-only machines.
"""

import inspect

import torch

import eval_model


def _device_param(fn):
    params = inspect.signature(fn).parameters
    assert "device" in params, f"{fn.__name__} must accept a device"
    return params["device"]


def test_eval_helpers_auto_select_the_gpu():
    for fn in (eval_model.per_channel_mae,
               eval_model.calibration_stats,
               eval_model.input_sensitivity):
        default = _device_param(fn).default
        # A sensible default is either "auto" or None -- both mean "pick for me".
        assert default in ("auto", None), (
            f"{fn.__name__} defaults to device={default!r}, which pins evaluation "
            "to the CPU and makes a full-val run impractically slow"
        )


def test_explicit_cpu_is_still_honoured():
    """Tests and CPU-only machines must keep working."""
    class _DS:
        def __len__(self): return 1
        def __getitem__(self, i):
            return torch.zeros(2, 3, 4, 4), torch.zeros(2, 3)

    class _Zero(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], x.shape[1], 3)

    mae = eval_model.per_channel_mae(_Zero(), _DS(), device="cpu")
    assert mae["steer"] == 0.0


def test_auto_resolves_to_cuda_when_available():
    resolved = eval_model.resolve_device("auto")
    if torch.cuda.is_available():
        assert resolved.startswith("cuda")
    else:
        assert resolved == "cpu"
