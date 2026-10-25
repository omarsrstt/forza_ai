"""Shared preprocessing for behavioural cloning.

Single canonical place where recorded gamepad events and screen frames are
converted into the physical labels and normalised tensors the model consumes.
Both training and the live driver import from here so they cannot drift apart.
"""

from typing import Callable

import cv2
import numpy as np
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
