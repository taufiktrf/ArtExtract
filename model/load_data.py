from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image
import os
import torch


class UNetDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        
        # Ensure each image has corresponding 8 masks
        self.masks = {img_name: sorted([f for f in os.listdir(masks_dir) if f.startswith(img_name.split('_RGB')[0])]) for img_name in self.images}
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load all 8 masks
        mask_paths = [os.path.join(self.masks_dir, mask_name) for mask_name in self.masks[img_name]]
        masks = [Image.open(mask_path) for mask_path in mask_paths]
        
        if self.transform:
            image = self.transform(image).float()
            masks = [self.transform(mask).float() for mask in masks]
        
        # Stack masks into a single tensor with shape (8, H, W)
        masks = torch.stack(masks)
        return image, masks

def load_datasets(train_path, val_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  
        transforms.ToTensor(),        
    ])
    
    train_images_dir = train_path + 'rgb_images/'
    train_masks_dir = train_path + 'ms_masks/'
    val_images_dir = val_path + 'rgb_images/'
    val_masks_dir = val_path + 'ms_masks/'
    
    # Create custom datasets
    train_dataset = UNetDataset(images_dir=train_images_dir, masks_dir=train_masks_dir, transform=transform)
    val_dataset = UNetDataset(images_dir=val_images_dir, masks_dir=val_masks_dir, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader