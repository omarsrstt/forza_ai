# Forza Behavioural Cloning — v2 Design

Date: 2026-10-25
Status: approved (in chat) — supersedes the original `train_model.py` / `drive.py` design

## Intent

Make the recorded dataset actually train a model that learns to drive, and make the
resulting model actually load and drive the car. The original pipeline trained a model
that collapsed to a constant output, and — even then — `drive.py` never loaded the
trained weights. This design fixes the data/loss/train-inference pipeline end to end.

## Goals

- A model whose output **depends on the input image** (not a constant).
- Training that reports an honest, physical-space validation metric on **held-out sequences**.
- Inference that feeds the model the **same kind of input it was trained on**.
- Everything must fit an **8 GB VRAM** budget (RTX 3070 Laptop).

## Non-goals (this iteration)

- No new data collection (user cannot record more).
- No DAgger / recovery data / closed-loop fine-tuning.
- No reinforcement learning.
- No new backbone research beyond the survey already done.

## Anti-goals (would be failure even if it "works")

- A model that drives by outputting the dataset mean.
- A validation metric that looks good while the model ignores the image.
- Silently loading zero weights (the original `strict=False` bug).

## Constraints

- Dataset: ~43,555 frames across 14 sequences, 224x224 JPG, RGB, labels per frame:
  `[L1, R1, Square, Circle, SteerLR, R2(throttle), L2(brake)]`, all in pygame's
  `[-1, 1]` range (trigger rest = -1).
- No train/val separation currently exists.
- Buttons are effectively dead in this data: LB 0.00%, RB 0.00%, Circle 0.00%,
  Square 0.02% pressed. LB/RB are gear down/up in manual mode; Square is probably
  handbrake. They are **unlearnable** from this dataset → dropped from the loss.

## Decisions (user-approved)

- **Backbone:** ConvNeXt-Tiny, ImageNet-pretrained. The 2024 "Which Backbone to Use"
  study finds pure CNNs (ConvNeXt) beat transformers in the low-data fine-tune regime;
  reusable across many driving projects. Keep it.
- **Temporal:** LSTM over per-frame features (Berkeley deep-RL notes; Waymo CNN+LSTM
  study). LSTM beats frame-stacking for fixed VRAM cost.
- **Supervision:** dense — predict at **every timestep**, not just the last.
- **Train/inference parity:** inference replays a rolling k-frame window identical to
  training input, instead of a stateful single-frame feed.
- **Outputs:** 3 physical channels — steering `tanh -> [-1,1]`, throttle `sigmoid -> [0,1]`,
  brake `sigmoid -> [0,1]`. This also fixes the vgamepad trigger range bug at the source.
- **Loss:** Huber on all 3 channels; metric = physical-space MAE (not logits).
- **Validation:** hold out **whole sequences** so no frame from a val sequence is in train.
- **Augmentation:** drop rotation/erasing (fights labels, breaks temporal consistency);
  keep only mild, sequence-consistent colour jitter.
- **Optimiser:** AdamW, discriminative LR (backbone low, head higher), plain cosine
  schedule (no warm restarts), grad clip 0.5, AMP `16-mixed`.
- **Builder's call:** freeze the first 2 ConvNeXt stages initially (low-data fine-tune
  recipe) — reversible; unfreeze later if val plateaus high.

## What's left to the builder

- Exact window length, batch size, grad-accumulation factor (choose from a VRAM probe).
- Whether a small frame-stacked fallback is worth keeping.
- Exact colour-jitter magnitudes.

## Resolved during builder check (2026-10-25)

- **Root cause of the v1 collapse** was a *training/forward bug*, not the raw data:
  the trained ConvNeXt produced a **constant** feature vector across all 150 images
  (std 0.0000) vs 0.268 for a fresh model, so the LSTM only ever saw the dataset mean.
  The data is driving-dominated (77.7% full throttle, 7.7% idle, longest idle run 36
  frames, no menu/countdown sections) and steering is symmetric around zero
  (median -0.004). So reweighting is a *nice-to-have*, not the missing fix.
- **Frame timing parity:** training frames were recorded at 15 FPS (`RECORD_FPS_CAP`),
  so inference will be **paced to ~15 FPS** too. To stay within the 15 FPS budget on
  8 GB, `drive.py` caches per-frame ConvNeXt features and only encodes the newest frame
  per step (LSTM runs over the cached feature window). Ring-buffer window = train window.
- **Train/val split:** hold out **3 of 14 sequences**, chosen to span steering diversity
  (high/low `|steer|>0.6` fraction) so val is representative. Split is persisted as a
  JSON file so train and eval agree.
- **Inference transmission:** the dropped buttons include gear up/down, so the model
  cannot shift. Constraint: **the car must be driven in automatic transmission (and
  default assists) at inference.** `drive.py` ships no gear logic. (Recorded ~0% LB/RB
  is consistent with an automatic gearbox during recording.)
- **Acceptance bar** for a training run: on the held-out val sequences,
  (a) physical-space MAE for steering must beat the constant-mean predictor by a clear
  margin, and (b) input-sensitivity (max abs output diff between a real window and a
  black window) must exceed 0.05 on steering and throttle. A run failing (b) is a
  collapse regardless of MAE.
- **No separate frame-stacked fallback** for now — windowed LSTM only; revisit only if
  LSTM train/live parity proves troublesome.

## Implementation steps

1. **Measurement harness** — checkpoint loader fix + `eval_model.py` with per-channel
   physical MAE and an input-sensitivity check, plus a sequence-level train/val split.
2. **Drive wiring fix** — BGR->RGB, trigger index un-swap, trigger range `(x+1)/2`,
   window-replay hidden handling, remove x10 / /5 hacks.
3. **Training v2** — dense supervision, 3 outputs, val split, physical metric, VRAM-sized
   batch + grad accumulation.
4. **Dry run** then **overnight run**, verified with the harness.
