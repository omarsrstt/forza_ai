import os
import re
import json
import wandb
from PIL import Image
from natsort import natsorted

import torch
import torch.nn as nn
import torch.nn as nn
from torchmetrics.regression import MeanAbsoluteError
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger

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
                 data_dir: str = "./dataset", 
                 transform = None,
                 sequence_length: int = 100,
                 stride_length: int = None):
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length # Number of frames in a sequence
        self.sequences = {}  # {sid: [(frame_id, img_filename, event_filename), ...]}

        # Regular expressions to extract sequence ID and frame ID
        img_pattern = re.compile(r'seq_(\d+)_image_\d+_(\d+)\.(jpg|png)')
        event_pattern = re.compile(r'seq_(\d+)_events_\d+_(\d+)\.json')

        # Store file names
        image_files = natsorted([f for f in os.listdir(self.data_dir) if f.endswith('.jpg') or f.endswith('.png')])
        event_files = natsorted([f for f in os.listdir(self.data_dir) if f.endswith('.json')])
        assert len(image_files) == len(event_files), "Number of images and event files must be the same"

        # Organize files by sequence ID
        img_files_by_sid = {} # {sid: [(frame_id, img_filename), ...]}
        for img_file in image_files:
            match = img_pattern.match(img_file)
            if match:
                sid, fid, _ = match.groups()
                if sid not in img_files_by_sid:
                    img_files_by_sid[sid] = []
                img_files_by_sid[sid].append((int(fid), img_file))

        event_files_by_sid = {} # {sid: [(frame_id, event_filename), ...]}
        for event_file in event_files:
            match = event_pattern.match(event_file)
            if match:
                sid, fid = match.groups()
                if sid not in event_files_by_sid:
                    event_files_by_sid[sid] = []
                event_files_by_sid[sid].append((int(fid), event_file))

        # Combine image and event files, ensuring matching frame IDs
        for sid in img_files_by_sid:
            if sid in event_files_by_sid:
                img_frames = dict(img_files_by_sid[sid]) # {frame_id: img_filename}
                event_frames = dict(event_files_by_sid[sid]) # {frame_id: event_filename}
                common_fids = sorted(set(img_frames.keys()) & set(event_frames.keys()))
                sequence = []
                for fid in common_fids:
                    img_filename = img_frames[fid]
                    event_filename = event_frames[fid]
                    sequence.append((fid, img_filename, event_filename))
                self.sequences[sid] = natsorted(sequence)
            else:
                print(f"Warning: No event files for sequence {sid}")

        # Generate samples with fixed sequence length
        self.samples = []  # Each sample is a list of (img_path, event_path)
        if stride_length is None:
            stride_length = max(1, sequence_length//2)
        for sid in self.sequences:
            frames = self.sequences[sid]
            num_frames = len(frames)
            if num_frames >= sequence_length:
                for i in range(0, num_frames - sequence_length + 1, stride_length):
                    sample_frames = frames[i:i+sequence_length]
                    self.samples.append(sample_frames)
            else:
                print(f"Sequence {sid} is shorter than the sequence length ({num_frames} frames). Skipping.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_frames = self.samples[idx]

        # Load all sample images and events
        images = []
        events = []

        for _, img_filename, event_filename in sample_frames:
            img_path = os.path.join(self.data_dir, img_filename)
            event_path = os.path.join(self.data_dir, event_filename)
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
            
            # Load events
            with open(event_path, 'r') as f:
                event = json.load(f)
            event = torch.tensor(event, dtype=torch.float32)
            events.append(event)

        # Stack images and events
        images = torch.stack(images)  # Shape: (sequence_length, C, H, W)
        events = torch.stack(events)  # Shape: (sequence_length, num_outputs)

        return images, events

# Define the ConvNeXt Tiny model for regression
class ConvNeXtTinyRegression(nn.Module):
    def __init__(self, num_outputs=7):
        super(ConvNeXtTinyRegression, self).__init__()
        self.model = models.convnext_tiny(weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # self.model.classifier
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_outputs)
    
    def forward(self, x):
        return self.model(x)


class ConvNextTinyLSTMRegression(nn.Module):
    def __init__(self, num_outputs=7):
        super(ConvNextTinyLSTMRegression, self).__init__()
        # Load the ConvNeXt model
        self.convnext = models.convnext_tiny(weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # Replace the final classification layer
        self.convnext.classifier = nn.Sequential(*list(self.convnext.classifier.children())[:-1])
        self.lstm = nn.LSTM(input_size=768, hidden_size=512, num_layers=1,  dropout=0.3, batch_first=True)
        self.fc = nn.Linear(512, num_outputs)

        # Initialize LSTM hidden state
        self.hidden_state = None

    def reset_hidden_state(self):
        # Reset hidden state at the start of a new sequence
        self.hidden_state = None
    
    def forward(self, x):
        # Check if processing a sequence (training) or a single frame (live)
        if x.dim() == 5:
            # Training mode: process full sequence
            # During training: x shape is (batch_size, seq_len, C, H, W)
            batch_size, seq_len, C, H, W = x.shape # Unpacking dimensions
            x = x.view(batch_size * seq_len, C, H, W)  # Reshape to combine batch and sequence dimensions
            convnext_out = self.convnext(x) # Apply CNN to all images at once
            convnext_out = convnext_out.view(batch_size, seq_len, -1) # Reshape to separate sequence and batch size
            lstm_out, _ = self.lstm(convnext_out) # Apply LSTM
            out = self.fc(lstm_out[:, -1, :]) # Take the last timestep's output
            return out
        else:
            # Live mode: process one frame at a time (autoregressive)
            # Extract features from the current frame
            convnext_out = self.convnext(x)  # (batch_size, num_features)
            convnext_out = convnext_out.unsqueeze(1)  # (batch_size, 1, num_features)
            
            # Pass through LSTM with previous hidden state
            lstm_out, self.hidden_state = self.lstm(convnext_out, self.hidden_state)
            
            # Get prediction
            out = self.fc(lstm_out[:, -1, :])  # (batch_size, num_outputs)
            return out

class ForzaLightning(L.LightningModule):
    def __init__(self, 
                 model):
        super(ForzaLightning, self).__init__()
        self.model = model
        self.criterion = torch.nn.MSELoss()

        # Save hyperparameters (if any)
        self.save_hyperparameters()

        # Initialize metrics
        self.mae = MeanAbsoluteError()

    def training_step(self, batch, batch_idx):
        images, controls = batch
        
        # Forward pass
        output_controls = self(images)

        # If using LSTM, reduce controls to match output_controls shape
        if isinstance(self.model, ConvNextTinyLSTMRegression):
            controls = controls[:, -1, :]  # Take the last timestep of the sequence
        
        # Split outputs and targets into boolean and continuous parts
        output_bool = output_controls[:, :4]  # First 4 outputs (boolean)
        output_cont = output_controls[:, 4:]  # Last 3 outputs (continuous)
        target_bool = controls[:, :4]
        target_cont = controls[:, 4:]

        # Loss calculation
        bool_loss = nn.functional.binary_cross_entropy_with_logits(output_bool, target_bool)
        # cont_loss = nn.functional.mse_loss(output_cont, target_cont)
        cont_loss = nn.functional.huber_loss(output_cont, target_cont)
        loss = bool_loss + cont_loss  # Combine losses        
        
        # Calculate metric
        mae = self.mae(output_controls, controls)

        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('mae', mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=20)
        return [optimizer], [scheduler]


def main():
    # Clear previous cache before initiating training
    torch.cuda.empty_cache()

    # Load the config file
    with open('config.json', 'r') as file:
        config = json.load(file)

    # Training consts
    SUPRESS_LOGS = config["suppress_logs"]
    NUM_EPOCHS = config["num_epochs"]

    # Define transformations for the images
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),  # Randomly erase 2-10% of the image
        # transforms.Resize((224, 224)),
        transforms.Normalize(mean=config["normalize_mean"],
                         std=config["normalize_std"]),
    ])

    # Create dataset and dataloader & load model for training
    data_dir = config["data_dir"]
    if config["LSTM"]:
        dataset = ForzaLSTMDataset(data_dir,
                                   transform=transform,
                                   sequence_length = config["sequence_length"],
                                   stride_length = config["stride_length"])
        model = ConvNextTinyLSTMRegression()
    else:
        dataset = ForzaDataset(data_dir, transform=transform)
        model = ConvNeXtTinyRegression()
    dataloader = DataLoader(dataset, 
                            batch_size=config["batch_size"], 
                            shuffle=config["shuffle"], 
                            num_workers=config["num_workers"],
                            persistent_workers=config["persistent_workers"],
                            )

    # Initialize WandB logger
    if not SUPRESS_LOGS:
        wandb_logger = WandbLogger(project=config["wandb_dir"],
                                   config = config)
    else:
        wandb_logger = None

    # Initialize the forza lightning module
    forzalightning = ForzaLightning(model=model)

    # Use lightning trainer for training
    trainer = L.Trainer(precision='16-mixed',
                        logger=wandb_logger,
                        max_epochs = NUM_EPOCHS,
                        gradient_clip_val = 0.5,
                        enable_checkpointing=config["enable_checkpointing"],
                        deterministic=config["deterministic"],
                        accelerator=config["accelerator"])
    
    trainer.fit(model=forzalightning, train_dataloaders=dataloader)

if __name__ == "__main__":
    main()