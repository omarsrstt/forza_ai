import os
import json
import wandb
from PIL import Image
from natsort import natsorted

import torch
import torch.nn as nn
import torch.nn as nn
from torchmetrics import Accuracy, MeanAbsoluteError
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
        
        events = torch.tensor(events, dtype=torch.float32)
        
        return image, events
    

class ForzaLSTMDataset(Dataset):
    def __init__(self, 
                 data_dir = "./dataset", 
                 transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.sequences = self.group_sequences()

    def group_sequences(self):
        # Natsort will order the files in the correct manner such that each image file will correspond with its event file
        image_files = natsorted([f for f in os.listdir(self.data_dir) if f.endswith('.jpg') or f.endswith('.png')])
        event_files = natsorted([f for f in os.listdir(self.data_dir) if f.endswith('.json')])
        assert len(image_files) == len(event_files), "Number of images and event files must be the same"

        sequences = {}
        for image_file, event_file in zip(image_files, event_files):
            # get sequence number
            seq_number = int(image_file.split("_")[1])
            if seq_number not in sequences: # If sequence doesn't exist, create new
                sequences[seq_number] = {"images": [],
                                         "events": []}
            sequences[seq_number]["images"].append(image_file)
            sequences[seq_number]["events"].append(event_file)

        return sequences    # return grouped sequences

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_number = list(self.sequences.keys())[idx]
        sequence_data = self.sequences[seq_number]

        # Load all sequence images
        images = []
        for image_name in sequence_data["images"]:
            image_path = os.path.join(self.data_dir, image_name)
            # Load image
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)

        events = []
        for event_name in sequence_data['events']:
            event_path = os.path.join(self.data_dir, event_name)
            with open(event_path, 'r') as f:
                event_data = json.load(f)
            events.append(event_data)
        
        return images, torch.tensor(events, dtype=torch.float32)


# Define the ConvNeXt Tiny model for regression
class ConvNeXtTinyRegression(nn.Module):
    def __init__(self, num_outputs=7):
        super(ConvNeXtTinyRegression, self).__init__()
        self.model = models.convnext_tiny(pretrained=True)
        # self.model.classifier
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_outputs)
    
    def forward(self, x):
        return self.model(x)


class ConvNextTinyLSTMRegression(nn.Module):
    def __init__(self, num_outputs=7):
        super(ConvNextTinyLSTMRegression, self).__init__()
        # Load the pre-trained ConvNeXt Tiny model
        self.convnext = models.convnext_tiny(pretrained=True)
        # Remove the final classification layer
        self.convnext.classifier = nn.Sequential(*list(self.convnext.classifier.children())[:-1])
        self.lstm = nn.LSTM(input_size=768, hidden_size=512, num_layers=1, batch_first=True)
        self.fc = nn.Linear(512, num_outputs)
    
    def forward(self, x):
        # Apply CNN
        batch_size, seq_len, C, H, W = x.shape
        # batch_size, C, H, W = x.shape

        convnext_out = []
        for t in range(seq_len): # each output of the series is appended
            out = self.convnext(x[:, t, :, :, :])
            out = out.view(batch_size, -1)
            convnext_out.append(out)
        convnext_out = torch.stack(convnext_out, dim=1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(convnext_out)
        
        # Apply final fully connected layer
        out = self.fc(lstm_out[:, -1, :])
        return out

class ForzaLightning(L.LightningModule):
    def __init__(self, 
                 model):
        super(ForzaLightning, self).__init__()
        self.model = model
        self.criterion = torch.nn.MSELoss()
        self.mae = MeanAbsoluteError()

    def training_step(self, batch, batch_idx):
        images, controls = batch
        
        # Forward pass
        output_controls = self(images)
        
        # Loss calculation
        loss = self.criterion(output_controls, controls)
        
        # Calculate metric
        mae = self.mae(output_controls, controls)

        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('mae', mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def forward(self, x):
        self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10)
        return [optimizer], [scheduler]


def main():
    # Training consts
    SUPRESS_LOGS = True

    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Create dataset and dataloader
    data_dir = 'dataset'
    dataset = ForzaDataset(data_dir, transform=transform)
    # dataset = ForzaLSTMDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


    # Initialized WandB logger
    if not SUPRESS_LOGS:
        wandb_logger = WandbLogger(project='forza-convnext-lstm')
    else:
        wandb_logger = None

    # Load model for training
    # model = ConvNeXtTinyRegression()
    model = ConvNextTinyLSTMRegression()

    # Use the forza lightning module
    forzalightning = ForzaLightning(model=model)

    # Use lightning trainer for training
    trainer = L.Trainer(precision='16-mixed',
                        logger=wandb_logger,
                        max_epochs = 10,
                        enable_checkpointing=True,
                        deterministic=True,
                        accelerator="cuda")
    
    trainer.fit(model=forzalightning, train_dataloaders=dataloader)


if __name__ == "__main__":
    main()