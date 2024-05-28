from metrics import FeatureLoss, PixelwiseLoss, BCELogitLoss, PSNR_metrics, RRMSE_metrics, SSIM_metrics
from torchvision.models.vgg import VGG16_Weights
from load_data import load_datasets
import torchvision.models as models
from utils import plot_images, viz_train
from datetime import datetime
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from unet import UNet
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

def train_test_model(model, train_path, val_path,optimizer, scheduler, device, num_epochs=100, patience=5):
    vgg_feature_extractor = VGGFeatureExtractor()
    model.to(device)
    vgg_feature_extractor.to(device)
    feature_loss = FeatureLoss(vgg_feature_extractor).to(device)
    # pixelwise_loss = PixelwiseLoss().to(device)
    # BCE_loss = BCELogitLoss().to(device)
    
    model.train()
    train_loader, val_loader = load_datasets(train_path, val_path)
    best_loss = float('inf')
    psnr_calc, rrmse_calc, ssim_calc = PSNR_metrics(), RRMSE_metrics(), SSIM_metrics()
    
    train_losses, train_rrmses, val_losses, val_rrmses = [], [], [], []
    
    for epoch in range(num_epochs):        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(images)
            batch_loss, batch_psnr, batch_rrmse, batch_ssim = 0.0, 0.0, 0.0, 0.0
            
            for i in range(images.size(0)):
                for j in range(images.size(1)):
                    output_image = output[i, j].unsqueeze(0)
                    target_image = masks[i, j]
                    if epoch==40:
                        plot_images(output_image, target_image, epoch, j)           
                    loss = feature_loss(output_image, target_image) 
                    # + pixelwise_loss(output_image,target_image)
                    batch_loss += loss
                    batch_psnr += psnr_calc(output_image, target_image).item()
                    batch_rrmse += rrmse_calc(output_image, target_image)
                    batch_ssim += ssim_calc(output_image, target_image).item()
                    
            NCT = images.size(0) * images.size(1)*len(train_loader)
            batch_loss /= NCT
            batch_rrmse /= NCT
            batch_psnr /= NCT
            batch_ssim /= NCT
            
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
        #pixel wise loss is not used now
        val_loss, val_psnr, val_rrmse, val_ssim = evaluate_model(model, val_loader, feature_loss, None, device, epoch)
        
        train_losses.append(batch_loss)
        train_rrmses.append(batch_rrmse)
        
        val_losses.append(val_loss)
        val_rrmses.append(val_rrmse)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train PSNR: {batch_psnr:.4f}, Val PSNR: {val_psnr:.4f}, Train RRMSE: {batch_rrmse:.4f}, Val RRMSE: {val_rrmse:.4f}, Train SSIM: {batch_ssim:.4f}, Val SSIM: {val_ssim:.4f}")

#         if val_loss < best_loss:
#             best_loss = val_loss
#             dir = './best_model/'
#             os.makedirs(dir, exist_ok=True)
#             best_model_path = f"{dir}best_model_epoch_{epoch+1}.pth"
#             torch.save(model.state_dict(), best_model_path)

#         if len(val_losses) > patience and val_loss > min(val_losses[-patience:]):
#             print(f"Early stopping at epoch {epoch+1}")
#             break
            
    print("Train/Test completed")
    viz_train(train_losses, val_losses, train_rrmses, val_rrmses)          
    # return best_model_path

def early_stop(val_losses, epoch, patience):
    if epoch > patience:
        best_val_loss = min(val_losses[:epoch])
        if val_losses[epoch] > best_val_loss:
            return True
    return False

def evaluate_model(model, val_loader, feature_loss, pixelwise_loss, device,epoch):
    model.eval()  # Set the model to evaluation mode
    val_loss, val_psnr, val_rrmse, val_ssim = 0.0, 0.0, 0.0, 0.0
    psnr_calc, rrmse_calc, ssim_calc = PSNR_metrics(), RRMSE_metrics(), SSIM_metrics()
    # pixelwise_loss = PixelwiseLoss().to(device)
    # BCE_loss = BCELogitLoss().to(device)
    
    with torch.no_grad():  # Disable gradient computation
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            output = model(images)
            for i in range(images.size(0)):
                for j in range(images.size(1)):
                    output_image = output[i, j].unsqueeze(0)
                    target_image = masks[i, j]
                    # plot_images(output_image, target_image, epoch,i)
                    feature_loss_val = feature_loss(output_image, target_image)
                    # + pixelwise_loss(output_image,target_image)
                    # pixel_loss_val = pixelwise_loss(output_image, target_image)
                    val_loss += feature_loss_val
                    # BCE_loss(output_image, target_image)
                    psnr = psnr_calc(output_image, target_image)
                    val_psnr += psnr.item()
                    val_rrmse += rrmse_calc(output_image, target_image)
                    val_ssim += ssim_calc(output_image, target_image)
                    #torch.sqrt(torch.mean((output_image - target_image) ** 2))
            NC = images.size(0) * images.size(1)
            val_loss += val_loss / NC
            val_psnr += val_psnr / NC
            val_rrmse += val_rrmse / NC
            val_ssim += val_ssim/NC
    
    # Compute the average loss over all batches
    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)
    val_rrmse /= len(val_loader)
    val_ssim /= len(val_loader)
    return val_loss, val_psnr, val_rrmse, val_ssim


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
