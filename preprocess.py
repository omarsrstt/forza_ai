"""Shared preprocessing for behavioural cloning.

Single canonical place where recorded gamepad events and screen frames are
converted into the physical labels and normalised tensors the model consumes.
Both training and the live driver import from here so they cannot drift apart.
"""

from typing import Callable

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Order of the 7 recorded gamepad values, as produced by gamepad.GamepadListener:
# [L1, R1, Square, Circle, SteerLR, R2, L2]
LABEL_COLUMNS: list[str] = ["L1", "R1", "Square", "Circle", "SteerLR", "R2", "L2"]

# ImageNet statistics, matching the ConvNeXt weights used by the model.
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)


def to_physical(events: np.ndarray) -> np.ndarray:
    """Convert a 7-element raw events vector to ``[steer, throttle, brake]``.

    Raw triggers arrive in ``[-1, 1]``; physical pedal values are ``[0, 1]``.
    Raw steer already spans ``[-1, 1]`` and passes through unchanged.
    """
    events = np.asarray(events, dtype=np.float32)
    steer = events[4]
    throttle = (events[5] + 1.0) / 2.0
    brake = (events[6] + 1.0) / 2.0
    return np.array(
        [
            np.clip(steer, -1.0, 1.0),
            np.clip(throttle, 0.0, 1.0),
            np.clip(brake, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def build_train_transform(color_jitter: bool = True) -> Callable:
    """Build the training image transform.

    Colour jitter is optional so validation and inference can use the same
    normalisation without augmentation.
    """
    steps: list[Callable] = []
    if color_jitter:
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD))
    return transforms.Compose(steps)


def rgb_from_bgra(frame: np.ndarray) -> np.ndarray:
    """Convert an ``(H, W, 4)`` BGRA or ``(H, W, 3)`` BGR frame to ``(H, W, 3)`` RGB.

    Screen captures (``mss``) are BGRA while recorded JPEGs decode as BGR, so
    both shapes must map onto the same channel order the model was trained on.
    """
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError(
            f"expected an (H, W, 3) BGR or (H, W, 4) BGRA frame, got shape {frame.shape}"
        )
    code = cv2.COLOR_BGRA2RGB if frame.shape[2] == 4 else cv2.COLOR_BGR2RGB
    return cv2.cvtColor(frame, code)


# --- Steering-offset augmentation -------------------------------------------
#
# Behavioural cloning on 14 recorded clips learns to drive the recorded lines and
# nothing else: measured on held-out sequences the model outputs ~63% of the
# steering the human applied (calibration slope 0.63) while reproducing the
# training clips almost exactly (slope 1.00). It has no examples of "I am off the
# racing line, here is the correction", so it understeers and then cannot recover.
#
# Translating the frame horizontally and shifting the label by the matching
# amount synthesises those recovery states from data we already have. This is the
# single-camera version of NVIDIA's left/right camera trick.
#
# Geometry: a lateral displacement ``d`` shifts the image by ``f*d/Z`` pixels and
# requires roughly ``atan(d/L)`` of steering, so to first order the required
# steer offset is *linear* in the pixel shift -- which is what
# ``steer_offset_for_shift`` encodes. The constant of proportionality (the gain)
# depends on camera FOV and lookahead, neither of which is recorded, so it is a
# tunable hyperparameter rather than a derived quantity.


def shift_image(image, shift_frac: float):
    """Translate ``image`` horizontally by ``shift_frac`` of its width.

    Positive shifts move image content to the **right**. Edges are reflected
    rather than padded with black, so a shifted frame contains no artificial
    border that the model could learn as a shortcut.
    """
    if shift_frac == 0.0:
        return image
    arr = np.asarray(image.convert("RGB"))
    height, width = arr.shape[:2]
    # cv2.warpAffine places source column s at destination s + M13, so a
    # rightward move of dx needs M13 = +dx. (Getting this backwards shifts the
    # image one way while the label moves the other, which teaches the model to
    # steer *away* from the road -- worse than no augmentation.)
    matrix = np.float32([[1.0, 0.0, shift_frac * width], [0.0, 1.0, 0.0]])
    shifted = cv2.warpAffine(
        arr, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return Image.fromarray(shifted)


def steer_offset_for_shift(shift_frac: float, gain: float) -> float:
    """Steering correction that re-centres a frame shifted by ``shift_frac``.

    If the world appears to the right of centre the car sits to the left of its
    target line and must steer right, so the offset has the sign of the shift.
    Clamped to the valid steer range.
    """
    return float(np.clip(gain * shift_frac, -1.0, 1.0))


def sample_shift(rng, max_frac: float) -> float:
    """Uniformly sample a shift magnitude in ``[-max_frac, +max_frac]``."""
    return float(rng.uniform(-max_frac, max_frac))
