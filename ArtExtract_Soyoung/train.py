import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.models.vgg import VGG16_Weights
import torchvision.models as models
from PIL import Image

from utils.metrics import MS_SSIMLoss, EvalMetrics
from utils.data import load_datasets
from utils.vizImg import plot_images, viz_train
from model import SimplyUNet
import warnings

warnings.filterwarnings("ignore")

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.features = vgg.features[:23] 
        
    def forward(self, x):
        return self.features(x)

def train_test_model(model, train_path, val_path, optimizer, scheduler, device,epochs):
    vgg_feature_extractor = VGGFeatureExtractor().to(device)
    model.to(device)
    ms_ssim_loss = MS_SSIMLoss().to(device)
    train_loader, val_loader = load_datasets(train_path, val_path)
    best_loss = float('inf')
    metrics = EvalMetrics(vgg_feature_extractor).to(device)
    losses, train_lpipses, val_lpipses = [], [], []
    train_psnrs, val_psnrs = [], []
    train_ssims, val_ssims = [], []
    scaler = GradScaler()
    accumulation_steps = 2
    
    best_val_lpips = float('inf')
    best_model_dir = "best_model"
    os.makedirs(best_model_dir, exist_ok=True)
    best_model_number = 0

    for epoch in range(epochs):
        model.train()
        epoch_psnr, epoch_lpips, epoch_ssim = 0.0, 0.0, 0.0
        temp_cnt = 0
        optimizer.zero_grad()

        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            with autocast():  # Mixed precision
                output = model(images)
                loss = 0.0
                for i in range(images.size(0)):
                    for j in range(images.size(1)):
                        output_image = output[i, j].unsqueeze(0)
                        target_image = masks[i, j]
                        temp_cnt += 1
                        if epoch % 3 == 0 and temp_cnt < 3:
                            plot_images(output_image, target_image, epoch, j)
                        output_image = F.normalize(output_image, dim=1)
                        target_image = F.normalize(output_image, dim=1)
                        loss += ms_ssim_loss(output_image, target_image)
                        psnr_value, lpips_value, ssim_value = metrics(output_image, target_image)
                        epoch_psnr += psnr_value.item()
                        epoch_lpips += lpips_value.item()
                        epoch_ssim += ssim_value.item()
                        del output_image, target_image
                loss /= (images.size(0) * images.size(1))
                loss /= accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            losses.append(loss.item())
            del loss
            torch.cuda.empty_cache()  

        NCT = len(train_loader) * images.size(0) * images.size(1)
        epoch_lpips /= NCT
        epoch_psnr /= NCT
        epoch_ssim /= NCT

        del images, masks
        torch.cuda.empty_cache()  

        with torch.no_grad():
            train_lpipses.append(epoch_lpips)
            train_psnrs.append(epoch_psnr)
            train_ssims.append(epoch_ssim)
            val_psnr, val_lpips, val_ssim = evaluate_model(model, val_loader, device, epoch, metrics)
            val_lpipses.append(val_lpips)
            val_psnrs.append(val_psnr)
            val_ssims.append(val_ssim)
            
            # Save the best model based on validation LPIPS and delete previous best model
            if val_lpips < best_val_lpips:
                best_val_lpips = val_lpips
                new_model_path = os.path.join(best_model_dir, f"best_model_{best_model_number}_{best_val_lpips:.4f}.pth")
                if best_model_number > 0:
                    old_model_path = os.path.join(best_model_dir, f"best_model_{best_model_number - 1}_{best_val_lpips:.4f}.pth")
                    if os.path.exists(old_model_path):
                        os.remove(old_model_path)
                torch.save(model.state_dict(), new_model_path)
                best_model_number += 1
                print(f"Best model saved with Validation LPIPS: {best_val_lpips:.4f}")


            print(f"Epoch [{epoch+1}/{epochs}] Train PSNR: {epoch_psnr:.4f}, Val PSNR: {val_psnr:.4f}, Train LPIPS: {epoch_lpips:.4f}, Val LPIPS: {val_lpips:.4f}, Train SSIM: {epoch_ssim:.4f}, Val SSIM: {val_ssim:.4f}")

    print("Train/Test completed")
    # Uncomment the below if you want to check the training progress
    # viz_train(losses, train_lpipses, val_lpipses, train_psnrs, val_psnrs, train_ssims, val_ssims)
    
# Evaluate model with the validation dataset
# Current Val dataset is limited (Multispectral dataset of paintings)
def evaluate_model(model, val_loader, device, epoch, metrics):
    model.eval()
    val_psnr, val_lpips, val_ssim = 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            for i in range(images.size(0)):
                for j in range(images.size(1)):
                    output_image = output[i, j].unsqueeze(0)
                    target_image = masks[i, j]
                    output_image = F.normalize(output_image, dim=1)
                    target_image = F.normalize(output_image, dim=1)
                    psnr_value, lpips_value, ssim_value = metrics(output_image, target_image)
                    val_psnr += psnr_value.item()
                    val_lpips += lpips_value.item()
                    val_ssim += ssim_value.item()
                    del output_image, target_image

    NCT = len(val_loader) * images.size(0) * images.size(1)
    val_psnr /= NCT
    val_lpips /= NCT
    val_ssim /= NCT
    
    del images, masks
    torch.cuda.empty_cache()
    return val_psnr, val_lpips, val_ssim

def gen_img(model, best_model_path, test_path, output_dir, device):
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_loader = load_datasets(train_path, test_path,False)
    os.makedirs(output_dir,exist_ok=True)
    for i, (imgs,_) in enumerate(test_loader):
        imgs = imgs.to(device)
        output = model(imgs)
        for j in range(output.size(1)):
            output_channel = output[:, j, :, :]  # Extract jth channel from the output
            img_name = str(output_dir).split('/')[-1]
            save_image(output_channel, os.path.join(img_name, f"_channel_{j}.png"))

def get_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a U-Net model")
    parser.add_argument('-tr', '--trainpath', type=str, required=True, help='Path to training images')
    parser.add_argument('-v', '--valpath', type=str, required=True, help='Path to validation images')
    parser.add_argument('-te', '--testpath', type=str, help='Path to test images for generating multispectral images')
    parser.add_argument('-o', '--outputpath', type=str, help='Path to save generated multispectral images')    
    parser.add_argument('-lr', '--learningRate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('-g', '--genimg', action='store_true', help='Generate multispectral images')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SimplyUNet().to(device)
    learning_rate = args.learningRate
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if device == 'cuda':
        torch.cuda.empty_cache()
    train_test_model(model, args.trainpath, args.valpath, optimizer, scheduler, device, epochs=args.epochs)
    
    if args.genimg:
        generate_images(model, args.best_model_path, args.testpath, args.outputpath, device)
        print('Image generation completed')