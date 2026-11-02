"""Simulate drive.py exactly: one-step self-feed across frames, sliding window.

drive.py never rolls out inside a window. It keeps `k` cached CNN features, runs
the LSTM once, reads only the LAST output, sends it to the gamepad, and feeds that
same output back as `prev` on the next frame. So the recurrence spans FRAMES, one
step at a time, not 24 steps inside one window.

That difference matters: a 24-step teacher-forced rollout compounds 24 errors,
while the driver only ever compounds across real frames, where the image changes
too and can pull the prediction back. Measuring the wrong one either flatters or
condemns the model for no reason.

Features are encoded once and cached, because encoding dominates the cost and is
identical for every checkpoint.
"""

import json
import os

import numpy as np
import torch
from PIL import Image

import eval_model
import preprocess
import splits
import train_model

CH = ("steer", "throttle", "brake")
FRAMES_PER_SEQ = 500
CACHE = "logs/_val_features_cache.npz"


def load_sequence_frames(data_dir, sids):
    """Consecutive frame filenames and physical labels for each val sequence."""
    import re
    from collections import defaultdict

    img = re.compile(r"seq_(\d+)_image_\d+_(\d+)\.(jpg|png)")
    evt = re.compile(r"seq_(\d+)_events_\d+_(\d+)\.json")
    by = defaultdict(lambda: {"img": {}, "evt": {}})
    for name in os.listdir(data_dir):
        m = img.match(name)
        if m:
            by[int(m.group(1))]["img"][int(m.group(2))] = name
            continue
        m = evt.match(name)
        if m:
            by[int(m.group(1))]["evt"][int(m.group(2))] = name

    out = {}
    for sid in sids:
        fids = sorted(set(by[sid]["img"]) & set(by[sid]["evt"]))
        rows = []
        for fid in fids:
            with open(os.path.join(data_dir, by[sid]["evt"][fid])) as f:
                lab = preprocess.to_physical(np.array(json.load(f), dtype=np.float32))
            rows.append((by[sid]["img"][fid], lab))
        out[sid] = rows
    return out


def encode_cached(data_dir, per_seq, device, k):
    """Encode each sequence's frames once; return {sid: (feats, labels)}."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        print(f"using cached features ({CACHE})")
        return {int(s): (z[f"f{s}"], z[f"l{s}"]) for s in z["sids"]}

    tf = preprocess.build_train_transform(color_jitter=False)
    store, sids = {}, []
    for sid, rows in per_seq.items():
        rows = rows[:FRAMES_PER_SEQ + k]
        feats, labels = [], []
        print(f"  encoding seq {sid}: {len(rows)} frames", flush=True)
        batch = []
        for i in range(0, len(rows), 32):
            chunk = rows[i:i + 32]
            batch = [tf(Image.open(os.path.join(data_dir, n)).convert("RGB"))
                     for n, _ in chunk]
            with torch.no_grad():
                f = model_encode(torch.stack(batch).to(device))
            feats.append(f.float().cpu().numpy())
            labels.extend(lab for _, lab in chunk)
        store[sid] = (np.concatenate(feats), np.stack(labels).astype(np.float32))
        sids.append(sid)

    np.savez_compressed(CACHE, sids=np.array(sids),
                        **{f"f{s}": store[s][0] for s in store},
                        **{f"l{s}": store[s][1] for s in store})
    print(f"cached to {CACHE}")
    return store


_ENCODER = None


def model_encode(x):
    return _ENCODER.encode(x)


def drive_rollout(model, feats, labels, device, k):
    """One-step self-feed across frames, exactly as drive.py does."""
    model.eval().to(device)
    feat_t = torch.from_numpy(feats).to(device)
    prev = torch.zeros(1, 1, 3, device=device)     # centred, stationary start
    outs = []
    with torch.no_grad():
        for t in range(k - 1, len(feat_t)):
            window = feat_t[t - k + 1:t + 1].unsqueeze(0)      # (1,k,768)
            # Exactly drive.py: one forward pass, take the LAST output, feed it
            # back as prev on the next frame. Using forward_closed_loop here would
            # sum all 24 in-window deltas instead of one step.
            out = model.forward_from_features(window, prev=prev)
            prev = out[:, -1:]
            outs.append(prev[0, 0].cpu().numpy())
    return np.stack(outs), labels[k - 1:]


def main():
    global _ENCODER
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open("config.json"))
    split = splits.load_or_create_split(cfg["split_file"], cfg["data_dir"])
    k = cfg["sequence_length"]

    model, _ = train_model.load_forza_checkpoint(
        "logs/checkpoints/epoch00-step500-v1.ckpt", num_outputs=3)
    _ENCODER = model.eval().to(device)
    per_seq = load_sequence_frames(cfg["data_dir"], [int(s) for s in split["val"]])
    store = encode_cached(cfg["data_dir"], per_seq, device, k)

    # persistence reference across the same frames actually scored
    p_err, cnt = np.zeros(3), 0
    for sid, (f, l) in store.items():
        if len(l) > k:
            p_err += np.abs(l[k:] - l[k - 1:-1]).sum(0)
            cnt += len(l) - k
    print(f"\n{'persistence':>28} " + " ".join(
        f"{c}={p_err[i]/cnt:.4f}" for i, c in enumerate(CH)))

    print(f"\n{'checkpoint':>28} {'mae_s':>7} {'mae_t':>7} {'mae_b':>7} "
          f"{'slope':>6} {'r2':>6}")
    for path in ("logs/checkpoints/epoch00-step500-v1.ckpt",
                 "logs/checkpoints/epoch01-step1000-v1.ckpt",
                 "logs/checkpoints/epoch09-step5000-v1.ckpt"):
        if not os.path.exists(path):
            continue
        m, _ = train_model.load_forza_checkpoint(path, num_outputs=3)
        preds, trues = [], []
        for sid, (f, l) in store.items():
            p, t = drive_rollout(m, f, l, device, k)
            preds.append(p)
            trues.append(t)
        pred = np.concatenate(preds)
        true = np.concatenate(trues)
        mae = np.abs(pred - true).mean(0)
        ts, ps = true[:, 0], pred[:, 0]
        slope = float(np.polyfit(ts, ps, 1)[0])
        r2 = float(1 - ((ts - ps) ** 2).sum() / ((ts - ts.mean()) ** 2).sum())
        print(f"{os.path.basename(path):>28} {mae[0]:7.4f} {mae[1]:7.4f} "
              f"{mae[2]:7.4f} {slope:6.3f} {r2:6.3f}")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
