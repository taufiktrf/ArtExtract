from metrics import FeatureLoss, PixelwiseLoss, MS_SSIMLoss, EvalMetrics
from torchvision.models.vgg import VGG16_Weights
from torchvision.models.alexnet import AlexNet_Weights
from load_data import load_datasets
import torchvision.models as models
from utils import plot_images, viz_train
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from unet_test import UNet
from PIL import Image
import numpy as np
import argparse
import torch
import os

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.features = vgg.features[:23] 
        
    def forward(self, x):
        return self.features(x)

def train_test_model(model, train_path, val_path, optimizer, scheduler, device, num_epochs=100, patience=5):
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

    for epoch in range(num_epochs):
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
            torch.cuda.empty_cache()  # Clear cache to free up memory

        NCT = len(train_loader) * images.size(0) * images.size(1)
        epoch_lpips /= NCT
        epoch_psnr /= NCT
        epoch_ssim /= NCT

        del images, masks
        torch.cuda.empty_cache()  # Clear cache to free up memory after epoch

        with torch.no_grad():
            train_lpipses.append(epoch_lpips)
            train_psnrs.append(epoch_psnr)
            train_ssims.append(epoch_ssim)
            val_psnr, val_lpips, val_ssim = evaluate_model(model, val_loader, device, epoch, metrics)
            val_lpipses.append(val_lpips)
            val_psnrs.append(val_psnr)
            val_ssims.append(val_ssim)

            print(f"Epoch [{epoch+1}/{num_epochs}] Train PSNR: {epoch_psnr:.4f}, Val PSNR: {val_psnr:.4f}, Train LPIPS: {epoch_lpips:.4f}, Val LPIPS: {val_lpips:.4f}, Train SSIM: {epoch_ssim:.4f}, Val SSIM: {val_ssim:.4f}")

    print("Train/Test completed")
    viz_train(losses, train_lpipses, val_lpipses, train_psnrs, val_psnrs, train_ssims, val_ssims)

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
        output = model(images)
        for j in range(output.size(1)):
            output_channel = output[:, j, :, :]  # Extract jth channel from the output
            img_name = str(output_dir).split('/')[-1]
            save_image(output_channel, os.path.join(img_name, f"_channel_{j}.png"))

def get_args():
    parser.add_argument('-tr', '--trainpath', type=str, help='train RGB image path')
    parser.add_argument('-v', '--valpath', type=str, help='validation RGB image path')
    parser.add_argument('-te', '--testpath', type=str, help='test RGB image path to generate multispectral images')
    parser.add_argument('-o', '--outputpath', type=str, help='Path to save generated multispectral images')    
    parser.add_argument('-l', '--learningRate', type=float, default=0.0002,help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=30,help='Number of epochs')
    parser.add_argument('-g', '--genimg', action='store_false', help='Generate multispectral images')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    unet_model = UNet()
    vgg_feature_extractor = VGGFeatureExtractor()

    unet_model.to(device)
    vgg_feature_extractor.to(device)
    pixelwise_loss = PixelwiseLoss().to(device)
    feature_loss = FeatureLoss(vgg_feature_extractor).to(device)
    
    learning_rate = args.learningRate
    train_path = args.trainpath
    val_path = args.valpath
    test_path = args.testpath
    output_dir = args.outputpath
    epochs = args.epochs
    gen_img = args.genimg
    
    #optimizer = optim.Adam(unet_model.parameters(), lr=learning_rate)
    # Following the paper using Adam optimizer with Nestrov momentum
    optimizer = optim.NAdam(unet_model.parameters(), lr=learning_rate)
    
    train_test_model(unet_model,train_path,val_path, feature_loss, pixelwise_loss, optimizer, device, num_epochs=epochs)
    print('train/test completed')
    
    if gen_img:
        gen_img(model, best_model_path, test_path, output_dir, device)
        print('image generation completed')
