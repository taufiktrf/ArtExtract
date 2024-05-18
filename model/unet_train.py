from unet_losses import FeatureLoss, PixelwiseLoss
from load_data import load_datasets
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
from torchvision.models.vgg import VGG16_Weights
from unet import UNet
import torch
from PIL import Image
import argparse
import os

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = vgg.features[:23] 

    def forward(self, x):
        return self.features(x)

def train_test_model(model, train_path, val_path, feature_loss, pixelwise_loss, optimizer, device, num_epochs=100, patience=5):
    model.train()
    train_loader, val_loader = load_datasets(train_path, val_path)
    best_loss = float('inf')
    best_model_path = None

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            output = model(images)
            
            total_loss = 0.0
            for i in range(8):  # Assuming 8 output channels
                output_image = output[:, i, :, :].unsqueeze(1)
                # Extract ith channel from the output (B,1,H,W)
                target_image = masks[:, i, :, :]  # Get corresponding target mask
                
                feature_loss_val = feature_loss(output_image, target_image)
                pixel_loss_val = pixelwise_loss(output_image, target_image)
                total_loss += feature_loss_val + pixel_loss_val
            
            total_loss /= 8  # Average over the 8 channels
            total_loss.backward()
            optimizer.step()     
            running_loss += total_loss.item()
        # Evaluate model on validation data
        print('evaluate_model')
        val_loss = evaluate_model(model, val_loader, feature_loss, pixelwise_loss, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            save_dir = './best_model/'
            os.makedirs(save_dir,exist_ok=True)
            best_model_path = f"{save_dir}best_model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), best_model_path)
        
        # Check for early stopping
        if early_stop(val_loss, epoch, patience):
            print(f"Early stopping- No improvement for {patience} epochs.")
            break
    print("Train/Test completed")
    return best_model_path

def early_stop(val_loss, epoch, patience):
    if epoch > patience:
        best_val_loss = min(val_loss[:epoch])
        if val_loss[epoch] > best_val_loss:
            return True
    return False

# Fix the evaluate model error!!!!
def evaluate_model(model, val_loader, feature_loss, pixelwise_loss, device):
    model.eval()
    with torch.no_grad():
        print('with statement')
        print(val_loader)
        for images, masks in val_loader:
            print('1')
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            output = model(images)
            
            total_loss = 0.0
            for i in range(8):  # Assuming 8 output channels
                print('2')
                output_image = output[:, i, :, :].unsqueeze(1)
                # Extract ith channel from the output (B,1,H,W)
                target_image = masks[:, i, :, :]  # Get corresponding target mask

                feature_loss_val = feature_loss(output_image, target_image)
                pixel_loss_val = pixelwise_loss(output_image, target_image)
                total_loss += feature_loss_val + pixel_loss_val
            
            total_loss /= 8  # Average over the 8 channels
            total_loss.backward()
            optimizer.step()     
            val_loss += total_loss.item()
    val_loss /= len(val_loader)  # Average over the validation set
    return val_loss

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
