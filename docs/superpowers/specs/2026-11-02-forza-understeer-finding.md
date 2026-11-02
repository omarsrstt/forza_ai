# Forza Behavioural Cloning — Why It Understeers

Date: 2026-11-02
Status: investigation closed
Supersedes: nothing. This closes out the open question in
`docs/superpowers/specs/2026-10-25-forza-bc-design.md` about whether the model
drives well, not whether it loads and runs correctly.

## Summary

The pipeline works. The model loads, responds to its input, and drives. It
understeers, and five separate fixes failed to change that. The cause is not the
loss, the head, the optimiser, the augmentation or the temporal model: **the
dataset does not contain the information needed to choose a steering magnitude.**
Steering depends on speed and yaw rate, and the recordings contain only controller
states — no telemetry of any kind.

The one-line version: on the measurement that predicts a real drive, the shipped
model is about 5% better than holding the wheel still, and roughly three times
worse than a policy that cannot exist (one that knows the human's last command).

## The measurement problem that had to be fixed first

Every earlier decision was made on **teacher-forced** error: the model is given
the human's true previous action and scored on its own output. The car never has
that. It only ever has its own last output. Teacher forcing re-anchors the model
to ground truth every frame, which hides compounding drift entirely.

The drive-accurate measurement (`_drive_sim.py`) replays exactly what `drive.py`
does:

- CNN features are encoded once per frame and cached, so a 500-frame rollout costs
  seconds rather than minutes;
- a rolling window of `k=24` cached features is fed to the model;
- only the **last** output is sent to the gamepad and fed back as the previous
  action on the next frame;
- the rollout starts stationary, as the car does when the script starts.

This distinction decided an architecture question (see "Residual mode" below). Any
future change to this project should be judged on closed-loop numbers.

## Evidence

### 1. Vision explains almost nothing that recent actions do not

Ridge probes on frozen ConvNeXt-Tiny features, trained on 11 clips, scored on the
3 held-out clips:

| predictor | steer MAE | slope | R² |
|---|---|---|---|
| repeat the previous **true** action | **0.132** | 0.888 | 0.785 |
| history only (actions at lags 1,2,4,8) | 0.163 | 0.775 | 0.808 |
| image only (frozen features, linear) | 0.286 | 0.542 | 0.515 |
| the trained ConvNeXt+LSTM | 0.275 | 0.650 | 0.528 |

A linear readout of frozen features reproduces the full network to within 0.004
MAE. The LSTM and the fine-tuned backbone stage were contributing nothing
measurable. A residual probe - fit history first, then ask how much of the
leftover error the image explains - gives **2.7%** of the residual variance.

Vision is not useless, but on this data it is nearly redundant with recent
actions, and it is nowhere near sufficient on its own.

### 2. Labels are clean; the jerkiness is real driving

The raw stream contains full-lock to centre transitions inside 1/15 s, which
looked like label corruption. It is not:

- only **2.5%** of frames deviate by more than 0.3 from a 3-tap running median;
- 0.8% exceed 0.6;
- `corr(raw, denoised) = 0.983`;
- the spikes concentrate in sequences 4 and 5, both **training** clips; the three
  validation clips are the cleanest in the dataset.

Denoising the labels would therefore have "fixed" the wrong half of the data and
bought about 0.014 MAE. Label noise is real but secondary and is not the cause of
understeer.

### 3. Speed is not recoverable from a single frame

Steering magnitude is a function of road geometry, speed and yaw rate. Two visually
identical corners taken at 60 and 140 km/h require different wheel angles. The
dataset records `[L1, R1, Square, Circle, SteerLR, R2, L2]` and nothing else, so
speed and yaw rate are simply absent. No amount of architecture work on a
single-frame-per-timestep input can recover them, and 11 clips cannot teach the
mapping even if it could.

This also explains the calibration numbers directly: on training clips slope is
1.001 (memorised), on held-out clips 0.630-0.650. The conditional mean of a quantity
the input cannot identify is a shrunk estimate, which is exactly what an
understeering car looks like.

### 4. Closed-loop: the shipped model is only marginally better than doing nothing

Same cached features, same stationary start, scored against human labels:

| policy | steer MAE | steer R² |
|---|---|---|
| hold the wheel still (do nothing) | 0.480 | −0.01 |
| **shipped regressor (`epoch07.ckpt`)** | **0.454** | **0.17** |
| v2 augmented, `epoch01` | 0.469 | 0.13 |
| residual mode, `epoch09` | 0.481 | 0.02 |
| residual mode, `epoch00` | 0.623 | −0.55 |
| *hold the human's previous action (not available to the car)* | *0.147* | *0.81* |

The shipped model is the best policy available, and it beats inaction by about 5%.
That is the honest ceiling of this dataset with this approach.

## What was tried and rejected, and why

| change | result | conclusion |
|---|---|---|
| Steering-offset augmentation (NVIDIA left/right-camera trick), shift ≤8% of width, label offset `2.5 × shift` | slope 0.650 → 0.644 over 10 epochs | Injecting label noise at the same magnitude as the signal it was meant to add. Also made throttle slope worse (0.714 → 0.690). |
| Real Huber loss (`delta=0.2` instead of torch's default `1.0`, which makes "Huber" plain MSE over `[-1,1]` targets) | brake sensitivity 0.064 → 0.413 (6.4×), brake slope 0.419 → 0.500 | Genuine win, and confirms the latent bug. Brake only; steering unaffected. |
| Longer temporal window (`k=15` → `24`) | no change to steering slope | The LSTM already saw enough frames; more of the same input does not add the missing variable. |
| Epoch ceiling 30 → 10 | val steer MAE flat from epoch 1; brake sensitivity decayed 0.27 → 0.014 across 30 epochs | Nothing to gain from longer runs. |
| Post-hoc gain correction | steer MAE 0.3008 → 0.3009 | The shrink is state-dependent; a constant gain fitted on train does not transfer. |
| Residual head: `action = clip(prev + 0.5·tanh(delta))` | teacher-forced slope 0.866, R² 0.763, MAE 0.153 — the **first** model to clear the calibration bar. Closed-loop 0.481, i.e. **worse than doing nothing** | Teacher forcing supplies the human's action as the anchor, so a small correction is scored against a correct anchor. Vision predicts the steering *level*; asking for an increment discards exactly what it can contribute. |
| Drift hypothesis for the residual failure | mean delta bias −0.0002, no runaway saturation | Refuted. The corrections are uninformative, not accumulating. |

Two accepted tests went the wrong way (augmentation, residual) and both were
rejected on measurement, not on argument. The augmentation answer was already
suspicious from a mechanism standpoint before it ran; the residual answer was not,
and only the closed-loop replay exposed it.

## What was kept

- `_drive_sim.py` — drive-accurate closed-loop replay with one-pass feature caching.
  This is the tool that should gate any future model change.
- `eval_model.persistence_baseline_mae` and the `steer_beats_persistence` check in
  `meets_bar`. The v1 model passed every earlier check while losing to a one-line
  predictor.
- `eval_model.resolve_device` / `_model_device`. Eval helpers defaulted to CPU, and a
  full-val pass exceeded 30 minutes — which is why earlier comparisons silently used
  200-window subsamples whose slopes wandered by ±0.05.
- `train_model.main(config_path)` plus a configurable `checkpoint_dir`. A dry run
  previously wrote into `logs/checkpoints` and clobbered `last.ckpt`.
- `ForzaLightning` teacher-forced loss path and `forward_closed_loop`, retained behind
  the `residual=False` default so residual mode cannot be re-enabled without
  re-running the closed-loop comparison.

## What would actually be needed to go faster

In rough order of expected value:

1. **Record with telemetry.** Capture speed (and ideally yaw rate or steering angle
   and throttle together). A speed input is the one change that directly supplies the
   missing variable. This is the only fix with a clear mechanism behind it.
2. **More track diversity, not more frames.** 11 training clips teach 11 racing
   lines. Additional clips on unseen tracks are worth more than more laps of the same
   ones.
3. **A backbone that models motion natively** (S3D/VideoMAE rather than a per-frame
   CNN plus LSTM). Road texture expanding radially is a visual proxy for speed, and
   an explicit multi-frame backbone may extract it where pooled per-frame features do
   not. Given that the frozen-feature probe capped out at R² 0.52 with 24 frames of
   history already available, the probability of a large win here is low.
4. **Closed-loop fine-tuning on a simulator or in-game reset loop.** Out of scope for
   behavioural cloning from a fixed dataset, but it is the standard escape from this
   exact failure mode.

## Reproducing the numbers

- Ridge probes: `_diag_probe.py`, `_diag_probe2.py`
- Label quality: `_diag_labels2.py`, `_diag_spikes.py`
- Motion and persistence: `_diag_motion.py`
- Closed-loop replay: `_drive_sim.py` (caches features to
  `logs/_val_features_cache.npz` on first run)
- Drift decomposition: `_diag_drift.py`
- Final comparison table: `_diag_verdict.py`

All of these write nothing except the feature cache and are safe to re-run.
