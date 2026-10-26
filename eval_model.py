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


def _windows(dataset):
    """Yield ``(images, labels)`` items one at a time.

    ``dataset`` is indexed explicitly rather than iterated: our datasets define
    ``__getitem__`` over a fixed sample list without raising ``IndexError`` (and
    the test doubles never do either), so ``for x, y in dataset`` would loop
    forever. Indexing is the only safe way to walk it.
    """
    for i in range(len(dataset)):
        yield dataset[i]


def per_channel_mae(model, loader, device="cpu"):
    """Mean absolute error per channel in physical units.

    ``loader`` is any object supporting ``len``/``__getitem__`` that yields
    ``(images (k,C,H,W), labels (k,3))``. Items are re-batched to
    ``(1,k,C,H,W)`` because the model expects a leading batch dimension.
    """
    abs_sum = torch.zeros(len(CHANNELS), dtype=torch.float64)
    count = 0
    with torch.no_grad():
        for images, labels in _windows(loader):
            images = images.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0)
            preds = model(images)
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


def input_sensitivity(model, val_dataset, device="cpu"):
    """Per-channel max absolute output change when the input is blanked.

    A model that ignores its input returns identical predictions for the real
    window and an all-zero window, so every channel reads exactly 0.0. Any
    healthy model must move by a visible margin.
    """
    real = val_dataset[0][0].unsqueeze(0).to(device)
    black = torch.zeros_like(real)
    with torch.no_grad():
        real_out = model(real)
        black_out = model(black)
    diff = (real_out - black_out).abs().amax(dim=(0, 1)).detach().cpu()
    return dict(zip(CHANNELS, diff.tolist()))


def _format(scores):
    return "{" + ", ".join(f"{name}: {value:.4f}" for name, value in scores.items()) + "}"


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
    model.eval()

    model_mae = per_channel_mae(model, val_dataset)
    baseline_mae = constant_baseline_mae(val_dataset)
    sensitivity = input_sensitivity(model, val_dataset)

    print(f"model MAE:      {_format(model_mae)}")
    print(f"baseline MAE:   {_format(baseline_mae)}")
    print(f"sensitivity:    {_format(sensitivity)}")

    beats_baseline = model_mae["steer"] < baseline_mae["steer"]
    responsive = (
        sensitivity["steer"] > SENSITIVITY_THRESHOLD
        and sensitivity["throttle"] > SENSITIVITY_THRESHOLD
    )
    passed = beats_baseline and responsive
    print(f"steer beats constant baseline: {beats_baseline}")
    print(
        f"input sensitivity > {SENSITIVITY_THRESHOLD} on steer and throttle: {responsive}"
    )
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
