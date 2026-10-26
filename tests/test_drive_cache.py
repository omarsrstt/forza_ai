import torch
from drive_cache import FrameWindow

def test_window_pads_until_full():
    w = FrameWindow(k=3)
    out = w.push(torch.ones(768))
    assert out.shape == (1,3,768)
    assert torch.all(out[:, 0] == 0)     # left padding
    assert torch.all(out[:, -1] == 1)

def test_window_rolls_oldest_out():
    w = FrameWindow(k=2)
    w.push(torch.ones(768))
    out = w.push(torch.full((768,), 2.0))
    assert torch.all(out[0,0] == 1.0) and torch.all(out[0,1] == 2.0)
    out = w.push(torch.full((768,), 3.0))
    assert torch.all(out[0,0] == 2.0) and torch.all(out[0,1] == 3.0)
