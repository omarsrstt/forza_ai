"""Rolling per-frame feature window for stateless LSTM inference.

Training feeds the model a window of ``k`` consecutive frames; the live driver
caches the CNN features of the last ``k`` frames and re-runs only the LSTM over
that window, so it never pays to re-encode history and holds no hidden state
between steps. Until the buffer has seen ``k`` frames it is left-padded with
zeros, i.e. oldest slot first and the just-pushed frame last.
"""

from collections import deque

import torch
from torch import Tensor


class FrameWindow:
    """Fixed-length rolling buffer of ``(768,)`` per-frame features.

    ``push`` appends the newest feature and returns a ``(1, k, 768)`` batch, the
    same window shape the model is trained on: oldest frame first, newest last,
    zero-filled on the left while the buffer is still filling up.
    """

    def __init__(self, k: int) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = k
        self._features: deque[Tensor] = deque(maxlen=k)

    def push(self, feature: Tensor) -> Tensor:
        """Append ``feature`` (768,) and return the window as ``(1, k, 768)``."""
        feature = feature.detach().reshape(-1)
        self._features.append(feature)
        padding = self.k - len(self._features)
        window = [torch.zeros_like(feature) for _ in range(padding)]
        window.extend(self._features)
        return torch.stack(window).unsqueeze(0)
