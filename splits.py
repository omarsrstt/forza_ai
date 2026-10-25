"""Dataset sequence discovery and a deterministic, persisted train/val split.

The recorder writes one JPEG and one events JSON per frame, both tagged with a
sequence id and a frame id. Training needs to know which sequences exist and
which of them are held out for validation, and the split must be stable across
runs so evaluation and training agree on it. This module is the single place
that decides both.
"""

import json
import os
import random
import re

# The events file is the source of truth: every recorded frame has one, and its
# name carries both the sequence id and the trailing frame id.
_EVENTS_RE = re.compile(r"seq_(\d+)_events_\d+_(\d+)\.json")


def discover_sequences(data_dir: str) -> dict[int, list[int]]:
    """Map each recorded sequence id to its sorted list of frame ids.

    Sequence and frame ids are parsed as integers so callers never have to
    worry about string ordering (``"10"`` sorting before ``"9"``).
    """
    seqs: dict[int, list[int]] = {}
    for name in os.listdir(data_dir):
        match = _EVENTS_RE.fullmatch(name)
        if match is None:
            continue
        sid, fid = int(match.group(1)), int(match.group(2))
        seqs.setdefault(sid, []).append(fid)
    for fids in seqs.values():
        fids.sort()
    return seqs


def make_split(seqs: dict[int, list[int]], val_count: int, seed: int) -> dict[str, list[int]]:
    """Split sequence ids into disjoint, sorted train and validation lists.

    Selection uses a seeded RNG over the sorted ids so the same inputs always
    yield the same split. ``val_count`` is clamped to the number of sequences,
    which keeps small datasets (or an over-large request) from crashing while
    still returning disjoint lists.
    """
    ids = sorted(seqs)
    val_count = min(max(val_count, 0), len(ids))
    val = random.Random(seed).sample(ids, val_count)
    val_set = set(val)
    train = [sid for sid in ids if sid not in val_set]
    return {"train": sorted(train), "val": sorted(val)}


def load_or_create_split(
    path: str, data_dir: str, val_count: int = 3, seed: int = 0
) -> dict[str, list[int]]:
    """Return the split stored at ``path``, creating and writing it if absent.

    Persisting the split keeps training and evaluation pinned to the same
    held-out sequences even as the dataset or defaults change.
    """
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    split = make_split(discover_sequences(data_dir), val_count, seed)
    with open(path, "w") as f:
        json.dump(split, f)
    return split
