"""Offline evaluation harness for the behavioural-cloning model.

Two independent questions are answered here, because v1 of this project failed
in a way that MAE alone could not see: the model collapsed to a constant output
that ignored its input entirely (real, black and noise frames produced
bit-identical predictions) while still reporting a respectable-looking MAE.

  * ``per_channel_mae`` / ``constant_baseline_mae`` — physical-space accuracy on
    held-out sequences, compared against the trivial constant-mean predictor.
  * ``input_sensitivity`` — does the model's output actually change when its
    input changes at all? A constant-output model scores exactly 0.0 here.

``main`` prints all three and a PASS/FAIL verdict against the spec's
acceptance bar.
"""

import json

import torch

import preprocess
import splits
import train_model

# Output channel order used by the model and the physical labels.
CHANNELS = ("steer", "throttle", "brake")

# Spec acceptance bar: a run that fails the sensitivity check is a collapse
# regardless of its MAE, so both thresholds must be cleared.
SENSITIVITY_THRESHOLD = 0.05

# Calibration bar for steering. MAE and R2 are both blind to a shrunken
# predictor: a model that always outputs half the demanded lock is perfectly
# correlated (R2 = 1.0) and beats the constant baseline, yet it understeers off
# every corner. The measured on-car failure was slope 0.63 against slope 1.00 on
# training clips, so the slope is what has to be gated.
SLOPE_MIN = 0.85
R2_MIN = 0.75


def resolve_device(device="auto"):
    """Turn ``"auto"`` into a concrete device.

    Evaluation is unusably slow on the CPU (a full-val pass exceeded 30 minutes),
    so callers should not have to remember to ask for the GPU.
    """
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _model_device(model, device):
    """Resolve a device guaranteed to match the model's parameters.

    With ``"auto"`` we follow the model: an explicitly CPU-resident model is
    never handed CUDA inputs. ``main`` moves the model to the GPU first, which is
    what makes the default fast there without surprising every other caller.
    """
    if device not in (None, "auto"):
        return device
    params = getattr(model, "parameters", None)
    if callable(params):
        for p in params():
            return p.device
    return resolve_device("auto")


def _windows(dataset):
    """Yield ``(images, labels)`` items one at a time.

    ``dataset`` is indexed explicitly rather than iterated: our datasets define
    ``__getitem__`` over a fixed sample list without raising ``IndexError`` (and
    the test doubles never do either), so ``for x, y in dataset`` would loop
    forever. Indexing is the only safe way to walk it.
    """
    for i in range(len(dataset)):
        yield dataset[i]


def _prev_for(model, labels):
    """Teacher-forced previous action, or ``None`` for pre-residual models."""
    if getattr(model, "residual", False):
        return train_model.shift_labels(labels.unsqueeze(0))
    return None


def _predict(model, images, prev):
    """Call the model with ``prev`` only when it takes one.

    Test doubles and pre-residual checkpoints have a plain ``forward(x)``, so
    unconditionally passing a keyword would make this harness unusable for them.
    """
    if prev is None:
        return model(images)
    return model(images, prev=prev)


def per_channel_mae(model, loader, device="auto"):
    """Mean absolute error per channel in physical units.

    ``loader`` is any object supporting ``len``/``__getitem__`` that yields
    ``(images (k,C,H,W), labels (k,3))``. Items are re-batched to
    ``(1,k,C,H,W)`` because the model expects a leading batch dimension.
    """
    abs_sum = torch.zeros(len(CHANNELS), dtype=torch.float64)
    count = 0
    device = _model_device(model, device)
    with torch.no_grad():
        for images, labels in _windows(loader):
            images = images.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0)
            preds = _predict(model, images, _prev_for(model, labels.squeeze(0).to(device)))
            # Follow the model's device so a CPU test double is not asked to
            # subtract a GPU tensor from a CPU one.
            labels = labels.to(preds.device)
            abs_sum += (preds - labels).abs().sum(dim=(0, 1)).detach().cpu().double()
            count += preds.shape[0] * preds.shape[1]
    if count == 0:
        return {name: 0.0 for name in CHANNELS}
    mae = abs_sum / count
    return dict(zip(CHANNELS, mae.tolist()))


def constant_baseline_mae(dataset):
    """Per-channel MAE of the constant-mean predictor.

    The predictor is the per-channel mean of the held-out labels, so this is the
    score the model must beat to prove it learned anything beyond the label
    distribution.
    """
    windows = [
        labels.reshape(-1, len(CHANNELS)).double() for _, labels in _windows(dataset)
    ]
    if not windows:
        return {name: 0.0 for name in CHANNELS}
    stacked = torch.cat(windows, dim=0)  # (N, 3)
    mean = stacked.mean(dim=0)
    mae = (stacked - mean).abs().mean(dim=0)
    return dict(zip(CHANNELS, mae.tolist()))


def input_sensitivity(model, val_dataset, device="auto"):
    """Per-channel max absolute output change when the input is blanked.

    A model that ignores its input returns identical predictions for the real
    window and an all-zero window, so every channel reads exactly 0.0. Any
    healthy model must move by a visible margin.
    """
    device = _model_device(model, device)
    real = val_dataset[0][0].unsqueeze(0).to(device)
    black = torch.zeros_like(real)
    real_prev = _prev_for(model, val_dataset[0][1].to(device))
    with torch.no_grad():
        real_out = _predict(model, real, real_prev)
        black_out = _predict(model, black, real_prev)
    diff = (real_out - black_out).abs().amax(dim=(0, 1)).detach().cpu()
    return dict(zip(CHANNELS, diff.tolist()))


def _format(scores):
    return "{" + ", ".join(f"{name}: {value:.4f}" for name, value in scores.items()) + "}"


def calibration_stats(model, loader, device="auto"):
    """Per-channel ``(slope, r2)`` of ``pred`` against ``true``.

    ``slope`` is the least-squares coefficient ``a`` in ``pred = a*true + b``.
    ``a < 1`` means the model systematically under-commands the channel: it moves
    in the right direction but not far enough. This is invisible to MAE (which
    mixes shrink with noise) and invisible to R2 (correlation is invariant to
    scaling), so it has to be measured separately. ``a == 0`` with ``r2 == 0``
    means the channel is constant.

    Returns ``{name: (slope, r2)}``.
    """
    preds, trues = [], []
    device = _model_device(model, device)
    with torch.no_grad():
        for images, labels in _windows(loader):
            prev = _prev_for(model, labels.to(device))
            pred = _predict(model, images.unsqueeze(0).to(device), prev)
            preds.append(pred.reshape(-1, len(CHANNELS)).detach().cpu().double())
            trues.append(labels.reshape(-1, len(CHANNELS)).double())
    if not preds:
        return {name: (0.0, 0.0) for name in CHANNELS}
    pred = torch.cat(preds, dim=0)  # (N, 3)
    true = torch.cat(trues, dim=0)  # (N, 3)

    stats = {}
    for i, name in enumerate(CHANNELS):
        t, p = true[:, i], pred[:, i]
        tc = t - t.mean()
        denom = float((tc * tc).sum())
        slope = float((tc * (p - p.mean())).sum()) / denom if denom > 0 else 0.0
        pc = p - p.mean()
        corr_denom = float(tc.norm() * pc.norm())
        corr = float((tc * pc).sum()) / corr_denom if corr_denom > 0 else 0.0
        stats[name] = (slope, corr * corr)
    return stats


def persistence_baseline_mae(dataset):
    """Per-channel MAE of repeating the previous action.

    This is the baseline the project actually shipped behind. It is not academic:
    the driver knows the command it last sent, so ``y[t] = y[t-1]`` is available
    at inference, and on the recorded data it scores 0.13 steer MAE against the
    v1 model's 0.28. Any acceptance bar that omits it can certify a model that is
    worse than doing nothing.

    Window starts have no predecessor inside the window, so the first frame of
    each window is skipped rather than scored against a fabricated zero.
    """
    preds, trues = [], []
    for _, labels in _windows(dataset):
        labels = labels.reshape(-1, len(CHANNELS)).double()
        if labels.shape[0] < 2:
            continue
        preds.append(labels[:-1])
        trues.append(labels[1:])
    if not preds:
        return {name: 0.0 for name in CHANNELS}
    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)
    mae = (pred - true).abs().mean(dim=0)
    return dict(zip(CHANNELS, mae.tolist()))


def meets_bar(model_mae, baseline_mae, sensitivity, calibration, persistence=None):
    """Decide PASS/FAIL. Pure, so the printed verdict and tests cannot drift.

    ``persistence`` is optional so existing callers keep working, but ``main``
    always supplies it: without it the bar cannot distinguish a model that
    learned something from one that is worse than holding the last command.
    """
    checks = {
        "steer_beats_baseline": model_mae["steer"] < baseline_mae["steer"],
        "steer_responsive": sensitivity["steer"] > SENSITIVITY_THRESHOLD,
        "throttle_responsive": sensitivity["throttle"] > SENSITIVITY_THRESHOLD,
        "steer_calibration": (
            calibration["steer"][0] >= SLOPE_MIN
            and calibration["steer"][1] >= R2_MIN
        ),
    }
    if persistence is not None:
        checks["steer_beats_persistence"] = model_mae["steer"] < persistence["steer"]
    return all(checks.values()), checks


def main():
    with open("config.json") as f:
        config = json.load(f)

    data_dir = config.get("data_dir", "dataset")
    snapshot_path = config.get("snapshot", "model.ckpt")
    split_path = config.get("split_file", "splits.json")
    sequence_length = config.get("sequence_length", 150)
    stride_length = config.get("stride_length", 10)

    transform = preprocess.build_train_transform(color_jitter=False)
    split = splits.load_or_create_split(split_path, data_dir)
    val_dataset = train_model.ForzaLSTMDataset(
        data_dir,
        transform=transform,
        sequence_length=sequence_length,
        stride_length=stride_length,
        split=split,
        mode="val",
    )

    model, _ = train_model.load_forza_checkpoint(snapshot_path, num_outputs=3)
    # Evaluation has to run on the GPU: a full-val pass on CPU is impractically
    # slow, and the device defaults in the helpers below are "auto" now.
    model.eval().to(resolve_device())

    model_mae = per_channel_mae(model, val_dataset)
    baseline_mae = constant_baseline_mae(val_dataset)
    persistence_mae = persistence_baseline_mae(val_dataset)
    sensitivity = input_sensitivity(model, val_dataset)
    calibration = calibration_stats(model, val_dataset)

    print(f"model MAE:      {_format(model_mae)}")
    print(f"baseline MAE:   {_format(baseline_mae)}")
    print(f"persistence MAE:{_format(persistence_mae)}")
    print(f"sensitivity:    {_format(sensitivity)}")
    print("calibration:    {" + ", ".join(
        f"{n}: slope={calibration[n][0]:.3f} r2={calibration[n][1]:.3f}" for n in CHANNELS
    ) + "}")

    passed, checks = meets_bar(
        model_mae, baseline_mae, sensitivity, calibration, persistence=persistence_mae
    )
    print(f"steer beats constant baseline: {checks['steer_beats_baseline']}")
    print(f"steer beats persistence baseline: {checks['steer_beats_persistence']}")
    print(
        f"input sensitivity > {SENSITIVITY_THRESHOLD} on steer and throttle: "
        f"{checks['steer_responsive'] and checks['throttle_responsive']}"
    )
    print(
        f"steer calibration slope >= {SLOPE_MIN} and R2 >= {R2_MIN}: "
        f"{checks['steer_calibration']}"
    )
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
