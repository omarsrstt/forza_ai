"""Tests for steering-offset augmentation.

The augmentation synthesises the recovery states the dataset does not contain:
"the car sits left of the road centre, here is the right-hand correction". It is
only legitimate if the image and the steering label move *together* -- a shift
with a stale label teaches the model to steer away from the road, which is
strictly worse than no augmentation at all. These tests pin that coupling and
its sign convention.

Convention (matches gamepad.GamepadListener axis 0 and preprocess.to_physical):
positive steer is RIGHT. If image content is translated right, the road appears
right of centre, meaning the car is left of centre and must steer RIGHT. So a
positive shift implies a positive steer offset.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

import preprocess
import train_model


def _stripe_image(size=64, col=32):
    """Black image with a single white column at ``col``."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, col, :] = 255
    return Image.fromarray(arr, mode="RGB")


def _brightest_col(image):
    arr = np.asarray(image.convert("RGB"))
    return int(arr[0, :, 0].argmax())


def test_shift_moves_content_right():
    out = preprocess.shift_image(_stripe_image(col=32), 0.25)
    assert _brightest_col(out) == pytest.approx(48, abs=1)


def test_shift_moves_content_left():
    out = preprocess.shift_image(_stripe_image(col=32), -0.25)
    assert _brightest_col(out) == pytest.approx(16, abs=1)


def test_shift_is_identity_at_zero():
    src = _stripe_image(col=32)
    out = preprocess.shift_image(src, 0.0)
    assert np.array_equal(np.asarray(out), np.asarray(src))


def test_shift_preserves_shape_and_dtype():
    out = preprocess.shift_image(_stripe_image(), -0.1)
    assert out.size == (64, 64)
    assert np.asarray(out).dtype == np.uint8


def test_shift_introduces_no_black_border():
    # Reflecting the border keeps a uniform image uniform. A constant (black)
    # border would create a stripe the model could latch onto as a spurious
    # "off-centre" cue that never exists at inference time.
    uniform = Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8), mode="RGB")
    out = np.asarray(preprocess.shift_image(uniform, 0.25))
    assert out.min() == 200


def test_steer_offset_sign_follows_shift():
    # Positive shift = car left of centre = steer right (positive).
    assert preprocess.steer_offset_for_shift(0.08, gain=2.5) == pytest.approx(0.20)
    assert preprocess.steer_offset_for_shift(-0.08, gain=2.5) == pytest.approx(-0.20)
    assert preprocess.steer_offset_for_shift(0.0, gain=2.5) == 0.0


def test_steer_offset_clamps_to_valid_range():
    assert preprocess.steer_offset_for_shift(0.5, gain=10.0) == pytest.approx(1.0)
    assert preprocess.steer_offset_for_shift(-0.5, gain=10.0) == pytest.approx(-1.0)


def _mk_dataset(tmp_path, per_seq=4, steer=0.0):
    d = tmp_path / "dataset"
    d.mkdir(exist_ok=True)
    for sid in (0, 1):
        for i in range(1, per_seq + 1):
            # Stripe at a fixed column so a shift is visible in every frame.
            _stripe_image(col=32).save(d / f"seq_{sid}_image_{sid}_{i}.jpg", quality=100)
            (d / f"seq_{sid}_events_{sid}_{i}.json").write_text(
                json.dumps([0, 0, 0, 0, steer, -1, -1])
            )
    return str(d)


def _ds(data_dir, mode, shift_max_frac=0.0, gain=2.5):
    return train_model.ForzaLSTMDataset(
        data_dir,
        transform=transforms.ToTensor(),
        sequence_length=3,
        stride_length=3,
        split={"train": [0], "val": [1]},
        mode=mode,
        shift_max_frac=shift_max_frac,
        steer_shift_gain=gain,
    )


def test_dataset_applies_one_shift_to_the_whole_window(tmp_path, monkeypatch):
    """A window is one simulated camera offset: every frame and every label in
    that window must move by the same amount, not independently per frame."""
    monkeypatch.setattr(preprocess, "sample_shift", lambda rng, max_frac: 0.125)
    ds = _ds(_mk_dataset(tmp_path), "train", shift_max_frac=0.25, gain=2.0)
    images, labels = ds[0]

    # steer 0.0 + 2.0 * 0.125 = 0.25 on every timestep
    assert torch.allclose(labels[:, 0], torch.full((3,), 0.25))
    # and the image content moved right by the same 12.5% in every frame
    cols = [int(images[t, 0].argmax(dim=1).float().mean()) for t in range(3)]
    assert len(set(cols)) == 1
    assert cols[0] == pytest.approx(32 + 0.125 * 64, abs=2)


def test_dataset_without_augmentation_leaves_steer_untouched(tmp_path):
    ds = _ds(_mk_dataset(tmp_path, steer=0.3), "train")     # shift_max_frac=0.0
    _, labels = ds[0]
    assert torch.allclose(labels[:, 0], torch.full((3,), 0.3))


def test_val_mode_never_augments_even_if_enabled(tmp_path, monkeypatch):
    """Validation and inference must see the true distribution. Guarding on mode
    as well as on the magnitude means a stray config value cannot silently
    corrupt the val split."""
    monkeypatch.setattr(preprocess, "sample_shift", lambda rng, max_frac: 0.25)
    ds = _ds(_mk_dataset(tmp_path, steer=0.3), "val", shift_max_frac=0.25)
    _, labels = ds[0]
    assert torch.allclose(labels[:, 0], torch.full((3,), 0.3))
