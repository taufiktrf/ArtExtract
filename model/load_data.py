from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import torch
import os

class UNetDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images =  [f for f in sorted(os.listdir(images_dir)) if f.endswith('RGB.bmp') or f.endswith('.png')]
        
        # Ensure each image has corresponding 8 masks
        self.masks = {img_name: sorted([f for f in os.listdir(masks_dir) if f.startswith(img_name.split('_RGB')[0])]) for img_name in self.images}
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        mask_names = self.masks[img_name]
        masks = []
        for mask_name in mask_names:
            mask_path = os.path.join(self.masks_dir, mask_name)
            mask = Image.open(mask_path)
            mode = mask.mode
    
        # If the mask image has more than one channel, convert it to grayscale
            if mode not in ['L','I']: #watercolor image from CAVE dataset is RGBA 4 channel that causes error
                # Convert the mask image to grayscale
                mask = mask.convert('L')
            if self.transform:
                mask = self.transform(mask).float()
            masks.append(mask)

        if self.transform:
            image = self.transform(image).float()
        masks = torch.stack(masks)
        return image, masks

def load_datasets(train_path, val_path):
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
        
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  
        transforms.ToTensor(),   
    ])
    
    train_images_dir = train_path + 'rgb_images/'
    train_masks_dir = train_path + 'ms_masks/'
    val_images_dir = val_path + 'rgb_images/'
    val_masks_dir = val_path + 'ms_masks/'
    
    # Create custom datasets
    train_dataset = UNetDataset(images_dir=train_images_dir, masks_dir=train_masks_dir, transform=train_transform)
    val_dataset = UNetDataset(images_dir=val_images_dir, masks_dir=val_masks_dir, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    return train_loader, val_loader