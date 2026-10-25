import json, torch
from PIL import Image
import train_model, preprocess, splits

def _mk(tmp_path, per_seq=6):
    d = tmp_path / "dataset"; d.mkdir(exist_ok=True)
    for sid in (0,1):
        for i in range(1, per_seq+1):
            Image.new("RGB",(8,8),(i*10,0,0)).save(d / f"seq_{sid}_image_{sid}_{i}.jpg")
            (d / f"seq_{sid}_events_{sid}_{i}.json").write_text(json.dumps([0,0,0,0,0,-1,-1]))
    return str(d)

def _ds(tmp_path, mode, k=3):
    d = _mk(tmp_path)
    sp = {"train":[0], "val":[1]}
    return train_model.ForzaLSTMDataset(d, preprocess.build_train_transform(False), k, k, sp, mode)

def test_window_shapes_and_dense_labels(tmp_path):
    ds = _ds(tmp_path, "train")
    img, lab = ds[0]
    assert img.shape[1:] == (3,8,8) and lab.shape == (3,3)
    assert torch.allclose(lab[:,1], torch.zeros(3))   # throttle rest -> 0
    assert torch.allclose(lab[:,2], torch.zeros(3))   # brake rest -> 0

def test_no_window_crosses_sequence_boundary(tmp_path):
    ds = _ds(tmp_path, "train")
    for sample in ds.samples:
        assert len({f[0].split("_")[1] for f in sample}) == 1   # one sid per window

def test_train_and_val_use_different_sequences(tmp_path):
    tr = set(_ds(tmp_path, "train").sequences.keys())
    va = set(_ds(tmp_path, "val").sequences.keys())
    assert tr.isdisjoint(va)

def test_sequence_shorter_than_window_is_skipped(tmp_path):
    d = _mk(tmp_path, per_seq=2)
    ds = train_model.ForzaLSTMDataset(d, preprocess.build_train_transform(False), 5, 5, {"train":[0,1],"val":[]}, "train")
    assert len(ds) == 0
