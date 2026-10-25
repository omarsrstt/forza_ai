import numpy as np
from torchvision import transforms

import preprocess

def test_to_physical_maps_rest_triggers_to_zero():
    ev = np.array([0,0,0,0,-0.5,-1.0,-1.0], dtype=np.float32)
    out = preprocess.to_physical(ev)
    assert out.shape == (3,)
    assert np.isclose(out[0], -0.5)      # steer unchanged
    assert np.isclose(out[1], 0.0)       # throttle rest
    assert np.isclose(out[2], 0.0)       # brake rest

def test_to_physical_maps_full_triggers_to_one():
    ev = np.array([0,0,0,0, 0.0, 1.0, 1.0], dtype=np.float32)
    out = preprocess.to_physical(ev)
    assert np.isclose(out[1], 1.0) and np.isclose(out[2], 1.0)

def test_to_physical_clamps():
    ev = np.array([0,0,0,0, 5.0, 5.0, -5.0], dtype=np.float32)
    out = preprocess.to_physical(ev)
    assert out[0] == 1.0 and out[1] == 1.0 and out[2] == 0.0

def test_rgb_from_bgra_swaps_channels():
    frame = np.zeros((2,2,4), dtype=np.uint8)
    frame[...,0] = 10   # B
    frame[...,1] = 20   # G
    frame[...,2] = 30   # R
    out = preprocess.rgb_from_bgra(frame)
    assert out.shape == (2,2,3)
    assert out[0,0,0] == 30 and out[0,0,2] == 10

def test_rgb_from_bgra_accepts_three_channel_bgr():
    frame = np.zeros((2,2,3), dtype=np.uint8)
    frame[...,0] = 10   # B
    frame[...,2] = 30   # R
    out = preprocess.rgb_from_bgra(frame)
    assert out.shape == (2,2,3)
    assert out[0,0,0] == 30 and out[0,0,2] == 10

def test_label_columns_match_recorded_order():
    assert preprocess.LABEL_COLUMNS == ["L1","R1","Square","Circle","SteerLR","R2","L2"]

def test_build_train_transform_has_no_rotation_or_erasing():
    jittered = preprocess.build_train_transform(True)
    names = [type(step).__name__ for step in jittered.transforms]
    assert names == ["ColorJitter", "ToTensor", "Normalize"]
    plain = preprocess.build_train_transform(False)
    assert [type(step).__name__ for step in plain.transforms] == ["ToTensor", "Normalize"]
    banned = (transforms.RandomRotation, transforms.RandomErasing)
    assert not any(isinstance(step, banned) for step in jittered.transforms)
