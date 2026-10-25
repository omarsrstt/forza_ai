import json, os
import splits

def _mk(tmp_path):
    d = tmp_path / "dataset"; d.mkdir()
    for sid, n in [(0,5),(1,6),(2,7),(3,8)]:
        for i in range(1, n+1):
            (d / f"seq_{sid}_image_{sid}_{i}.jpg").write_bytes(b"x")
            (d / f"seq_{sid}_events_{sid}_{i}.json").write_text("[0,0,0,0,0,-1,-1]")
    return str(d)

def test_discover_sequences(tmp_path):
    seqs = splits.discover_sequences(_mk(tmp_path))
    assert set(seqs) == {0,1,2,3}
    assert seqs[0] == [1,2,3,4,5]

def test_make_split_disjoint_and_sized():
    seqs = {i: list(range(1,10)) for i in range(14)}
    sp = splits.make_split(seqs, val_count=3, seed=0)
    assert len(sp["val"]) == 3 and len(sp["train"]) == 11
    assert set(sp["train"]).isdisjoint(sp["val"])

def test_make_split_deterministic():
    seqs = {i: list(range(1,10)) for i in range(14)}
    assert splits.make_split(seqs, 3, 0) == splits.make_split(seqs, 3, 0)

def test_load_or_create_split_persists(tmp_path):
    d = _mk(tmp_path); p = str(tmp_path / "splits.json")
    a = splits.load_or_create_split(p, d, val_count=1, seed=0)
    assert os.path.exists(p)
    b = splits.load_or_create_split(p, d, val_count=1, seed=0)
    assert a == b

def test_load_or_create_split_round_trips_json(tmp_path):
    d = _mk(tmp_path); p = str(tmp_path / "splits.json")
    a = splits.load_or_create_split(p, d, val_count=1, seed=0)
    assert json.loads(open(p).read()) == a

def test_make_split_val_count_at_least_num_sequences_does_not_crash():
    seqs = {i: [1] for i in range(3)}
    sp = splits.make_split(seqs, val_count=5, seed=0)
    assert set(sp["val"]).isdisjoint(sp["train"])
    assert sorted(sp["val"] + sp["train"]) == [0, 1, 2]
