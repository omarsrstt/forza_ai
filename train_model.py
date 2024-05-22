import os
import json
import torch
from PIL import Image
import torch.nn as nn
from natsort import natsorted
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

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
    

# Define the ConvNeXt Tiny model for regression
class ConvNeXtTinyRegression(nn.Module):
    def __init__(self, num_outputs=7):
        super(ConvNeXtTinyRegression, self).__init__()
        self.model = models.convnext_tiny(pretrained=True)
        # self.model.classifier
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_outputs)
    
    def forward(self, x):
        return self.model(x)


def main():
    # # Define transformations for the images
    # transform = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor()
    # ])

    # # Create dataset and dataloader
    # data_dir = 'data'  # Replace with your data directory
    # dataset = ForzaDataset(data_dir, transform=transform)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Load model for training
    model = ConvNeXtTinyRegression()

    # # Example of iterating through the dataloader
    # for images, events in dataloader:
    #     print(images.shape)  # (batch_size, 3, 64, 64)
    #     print(events.shape)  # (batch_size, 7)
    #     # Your training code here


if __name__ == "__main__":
    main()