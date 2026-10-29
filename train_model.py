import os
import re
import json
import numpy as np
import wandb
from PIL import Image
from natsort import natsorted

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

import preprocess
import splits

class ForzaDataset(Dataset):
    def __init__(self, 
                 data_dir = "./dataset", 
                 transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = natsorted([f for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png')])
        self.event_files = natsorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
        
        assert len(self.image_files) == len(self.event_files), "Number of images and event files must be the same"
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        event_name = self.event_files[idx]
        
        img_path = os.path.join(self.data_dir, img_name)
        event_path = os.path.join(self.data_dir, event_name)
        
        # Load image
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        
        # Load events
        with open(event_path, 'r') as f:
            events = json.load(f) 
        events = torch.tensor(events, 
                              dtype=torch.float32,
                              requires_grad=False)
        
        return image, events
    

class ForzaLSTMDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 transform,
                 sequence_length: int,
                 stride_length: int,
                 split: dict,
                 mode: str):
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length  # Number of frames in a window
        self.stride_length = stride_length
        self.sequences = {}  # {int_sid: [(int_fid, img_filename, event_filename), ...]}

        # Regular expressions to extract sequence ID and frame ID
        img_pattern = re.compile(r'seq_(\d+)_image_\d+_(\d+)\.(jpg|png)')
        event_pattern = re.compile(r'seq_(\d+)_events_\d+_(\d+)\.json')

        # Store file names
        image_files = natsorted([f for f in os.listdir(self.data_dir) if f.endswith('.jpg') or f.endswith('.png')])
        event_files = natsorted([f for f in os.listdir(self.data_dir) if f.endswith('.json')])
        assert len(image_files) == len(event_files), "Number of images and event files must be the same"

        # Organize files by sequence ID
        img_files_by_sid = {}  # {int_sid: [(int_fid, img_filename), ...]}
        for img_file in image_files:
            match = img_pattern.match(img_file)
            if match:
                sid, fid, _ = match.groups()
                img_files_by_sid.setdefault(int(sid), []).append((int(fid), img_file))

        event_files_by_sid = {}  # {int_sid: [(int_fid, event_filename), ...]}
        for event_file in event_files:
            match = event_pattern.match(event_file)
            if match:
                sid, fid = match.groups()
                event_files_by_sid.setdefault(int(sid), []).append((int(fid), event_file))

        # Combine image and event files, keeping only sequences held out for this mode
        selected_sids = {int(sid) for sid in split[mode]}
        for sid in img_files_by_sid:
            if sid not in selected_sids:
                continue
            if sid in event_files_by_sid:
                img_frames = dict(img_files_by_sid[sid])  # {int_fid: img_filename}
                event_frames = dict(event_files_by_sid[sid])  # {int_fid: event_filename}
                common_fids = sorted(set(img_frames.keys()) & set(event_frames.keys()))
                sequence = []
                for fid in common_fids:
                    img_filename = img_frames[fid]
                    event_filename = event_frames[fid]
                    sequence.append((fid, img_filename, event_filename))
                self.sequences[sid] = natsorted(sequence)
            else:
                print(f"Warning: No event files for sequence {sid}")

        # Generate windows, never crossing a sequence boundary
        self.samples = []  # Each sample is a list of (img_filename, event_filename)
        for sid in sorted(self.sequences):
            frames = self.sequences[sid]
            num_frames = len(frames)
            if num_frames < sequence_length:
                print(f"Sequence {sid} is shorter than the sequence length ({num_frames} frames). Skipping.")
                continue
            for i in range(0, num_frames - sequence_length + 1, stride_length):
                self.samples.append([(img, event) for _, img, event in frames[i:i + sequence_length]])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_frames = self.samples[idx]

        # Load all sample images and dense physical labels
        images = []
        labels = []

        for img_filename, event_filename in sample_frames:
            img_path = os.path.join(self.data_dir, img_filename)
            event_path = os.path.join(self.data_dir, event_filename)

            # Load image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)

            # Load events and convert to physical labels
            with open(event_path, 'r') as f:
                event = json.load(f)
            labels.append(preprocess.to_physical(np.array(event, dtype=np.float32)))

        # Stack images and labels
        images = torch.stack(images)  # Shape: (sequence_length, C, H, W)
        labels = torch.as_tensor(np.stack(labels), dtype=torch.float32)  # (sequence_length, 3)

        return images, labels

class ConvNextTinyLSTMRegression(nn.Module):
    def __init__(self, num_outputs=3, hidden_size=512):
        super(ConvNextTinyLSTMRegression, self).__init__()
        # Load the ConvNeXt model
        self.convnext = models.convnext_tiny(weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # Replace the final classification layer with LayerNorm2d + Flatten -> (N, 768) features
        self.convnext.classifier = nn.Sequential(*list(self.convnext.classifier.children())[:-1])
        # Freeze the first two stages of the backbone
        for p in self.convnext.features[:2].parameters():
            p.requires_grad = False
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_outputs)

    def encode(self, x):
        # x: (N, C, H, W) -> (N, 768)
        return self.convnext(x)

    def forward_from_features(self, feat):
        # feat: (B, k, 768) -> (B, k, num_outputs)
        lstm_out, _ = self.lstm(feat)          # (B, k, hidden_size)
        out = self.fc(lstm_out)                # (B, k, num_outputs)
        steer = torch.tanh(out[..., 0])
        throttle = torch.sigmoid(out[..., 1])
        brake = torch.sigmoid(out[..., 2])
        return torch.stack([steer, throttle, brake], dim=-1)

    def forward(self, x):
        # x: (B, k, C, H, W) -> (B, k, num_outputs), one prediction per timestep
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.encode(x).view(batch_size, seq_len, -1)  # (B, k, 768)
        return self.forward_from_features(features)

# Output channel order, matching preprocess.to_physical and the model's heads.
CHANNELS = ("steer", "throttle", "brake")


class ForzaLightning(L.LightningModule):
    """Dense-supervision wrapper around the ConvNeXt+LSTM regressor.

    v1 supervised only the final timestep, which let the LSTM collapse to the
    dataset mean while the loss went flat. Here every timestep of every window is
    scored with Huber loss on the three physical channels, so each prediction in
    the window gets a gradient.

    Every hyperparameter has a default so the module can be constructed as
    ``ForzaLightning(model)`` in tests; ``main`` passes the config values through.
    """

    def __init__(self,
                 model,
                 lr_backbone: float = 1e-4,
                 lr_head: float = 5e-4,
                 weight_decay: float = 1e-2,
                 max_epochs: int = 30):
        super(ForzaLightning, self).__init__()
        self.model = model
        self.lr_backbone = lr_backbone
        self.lr_head = lr_head
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        # The wrapped module is checkpointed as its own state_dict; keep it out of
        # the hyperparameters dumped into the checkpoint.
        self.save_hyperparameters(ignore=["model"])

        # Running sums for an exact epoch-level per-channel MAE. Accumulated in
        # float64 on CPU so the metric does not drift with the AMP/device dtype.
        self._val_abs_sum = torch.zeros(len(CHANNELS), dtype=torch.float64)
        self._val_count = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, controls = batch
        pred = self.model(images)  # (B, k, 3), one prediction per timestep
        # Dense supervision: reduce over every timestep and channel, not [:, -1].
        loss = F.huber_loss(pred, controls)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, controls = batch
        pred = self.model(images)  # (B, k, 3)
        abs_err = (pred.detach() - controls).abs()  # (B, k, 3)
        mae = abs_err.mean(dim=(0, 1))  # per-channel mean over batch and time

        self._val_abs_sum += abs_err.sum(dim=(0, 1)).cpu().double()
        self._val_count += pred.shape[0] * pred.shape[1]

        return {"mae": {name: mae[i].item() for i, name in enumerate(CHANNELS)}}

    def on_validation_epoch_end(self):
        if self._val_count == 0:
            return
        mae = self._val_abs_sum / self._val_count
        for i, name in enumerate(CHANNELS):
            self.log(f"val_mae_{name}", mae[i], prog_bar=(name == "steer"))
        self._val_abs_sum.zero_()
        self._val_count = 0

    def configure_optimizers(self):
        # Two groups with discriminative LRs: the frozen first two ConvNeXt stages
        # are skipped entirely, and the head trains faster than the backbone.
        backbone_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("convnext."):
                backbone_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.lr_backbone},
                {"params": head_params, "lr": self.lr_head},
            ],
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.max_epochs, 1)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def load_forza_checkpoint(
    path: str, num_outputs: int = 3, map_location="cpu"
) -> tuple[nn.Module, dict]:
    # weights_only=False is required: real Lightning .ckpt files contain non-tensor
    # objects (the old project checkpoint even pickles a model class). This is a
    # locally-trusted file, never load checkpoints from untrusted sources.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = ckpt["state_dict"]
    stripped = {k.split(".", 1)[1] if k.startswith("model.") else k: v
                for k, v in state_dict.items()}
    model = ConvNextTinyLSTMRegression(num_outputs=num_outputs)
    model.load_state_dict(stripped, strict=True)
    return model.eval(), ckpt.get("hyper_parameters", {})


def _cap_windows(dataset, max_windows):
    """Truncate a dataset to at most ``max_windows`` windows, in place.

    Used by the ``max_windows`` config knob to run a cheap dry run over a slice
    of the data. ``samples`` (the flat window list) and ``sequences`` (the
    per-sequence frames) are rebuilt together so they stay consistent, and whole
    sequences are only ever partially consumed, never reordered.
    """
    if max_windows is None:
        return
    sequence_length = dataset.sequence_length
    stride_length = dataset.stride_length
    kept_sequences = {}
    kept_samples = []
    offset = 0
    for sid in sorted(dataset.sequences):
        frames = dataset.sequences[sid]
        if len(frames) < sequence_length:
            continue
        remaining = max_windows - len(kept_samples)
        if remaining <= 0:
            break
        num_windows = (len(frames) - sequence_length) // stride_length + 1
        take = min(num_windows, remaining)
        # Keep only the frames the retained windows actually read, so a partial
        # take leaves no orphaned frames behind in ``sequences``.
        used = (take - 1) * stride_length + sequence_length
        kept_sequences[sid] = frames[:used]
        kept_samples.extend(dataset.samples[offset:offset + take])
        offset += num_windows
    dataset.sequences = kept_sequences
    dataset.samples = kept_samples


def main():
    # Clear previous cache before initiating training
    torch.cuda.empty_cache()

    # Load the config file
    with open('config.json', 'r') as file:
        config = json.load(file)

    data_dir = config["data_dir"]
    sequence_length = config["sequence_length"]
    stride_length = config["stride_length"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    # Persisted, whole-sequence train/val split so eval and training agree.
    split = splits.load_or_create_split(
        config["split_file"],
        data_dir,
        val_count=config["val_count"],
        seed=config["split_seed"],
    )

    # Train with mild colour jitter; validate on the clean transform.
    train_dataset = ForzaLSTMDataset(
        data_dir,
        transform=preprocess.build_train_transform(color_jitter=True),
        sequence_length=sequence_length,
        stride_length=stride_length,
        split=split,
        mode="train",
    )
    val_dataset = ForzaLSTMDataset(
        data_dir,
        transform=preprocess.build_train_transform(color_jitter=False),
        sequence_length=sequence_length,
        stride_length=stride_length,
        split=split,
        mode="val",
    )

    # Optional dry-run cap (config "max_windows"): train and validate on a
    # bounded slice of windows instead of the whole dataset.
    max_windows = config.get("max_windows")
    _cap_windows(train_dataset, max_windows)
    _cap_windows(val_dataset, max_windows)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    model = ConvNextTinyLSTMRegression(num_outputs=3)
    forzalightning = ForzaLightning(
        model=model,
        lr_backbone=config["lr_backbone"],
        lr_head=config["lr_head"],
        weight_decay=config["weight_decay"],
        max_epochs=config["num_epochs"],
    )

    # Weights & Biases is opt-in: it blocks on an interactive auth prompt, which
    # would stall an unattended overnight run. Default to plain local CSV logs.
    if config.get("use_wandb", False):
        logger = WandbLogger(project=config.get("wandb_dir", "forza-convnext-lstm"),
                             config=config)
    else:
        logger = CSVLogger("logs", name="forza-convnext-lstm")

    # Explicit checkpointing: the default callback under WandbLogger saved nothing
    # recoverable in the first 7 epochs of the 2026-10-27 run (which then died with
    # zero checkpoints), so pin the dir, save every epoch, and keep a rolling last.ckpt.
    ckpt_dir = os.path.join("logs", "checkpoints")
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch{epoch:02d}-step{step}",
        every_n_epochs=1,
        save_top_k=-1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    resume_path = None
    last = os.path.join(ckpt_dir, "last.ckpt")
    if config.get("resume_from_last", False) and os.path.exists(last):
        resume_path = last
        print(f"[resume] continuing from {last}")

    trainer = L.Trainer(
        accelerator=config["accelerator"],
        precision=config["precision"],
        accumulate_grad_batches=config["accumulate_grad_batches"],
        gradient_clip_val=0.5,
        max_epochs=config["num_epochs"],
        enable_checkpointing=config["enable_checkpointing"],
        logger=logger,
        num_sanity_val_steps=config.get("num_sanity_val_steps", 0),
        callbacks=[checkpoint_cb],
    )

    trainer.fit(model=forzalightning,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=resume_path)


if __name__ == "__main__":
    main()