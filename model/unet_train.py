from losses import FeatureLoss, PixelwiseLoss
from torch.utils.data import DataLoader
from unet import UNet
import argparse

#set the train/test folder that contain N number of bands for each train image.
#make it based on the CAVE image

def load_datasets(train_path, test_path):
    train_dataset = datasets.ImageFolder(root=train_path, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = datasets.ImageFolder(root=test_path, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

def train_test_model(model, train_path, test_path, feature_loss, pixelwise_loss, optimizer, device, num_epochs=100):
    model.train()
    train_loader, test_loader = load_datasets(train_path,test_path)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            output = model(images)
                            
            feature_loss_val = feature_loss(output, images)
            pixel_loss_val = pixelwise_loss(output, images)
            loss = feature_loss_val + pixel_loss_val
            loss.backward()
            optimizer.step()
            train_loss += loss.item()        
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                output = model(images)
                feature_loss_val = feature_loss(output, images)
                pixel_loss_val = pixelwise_loss(output, images)
                loss = feature_loss_val + pixel_loss_val
                running_test_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        
print("Finished Training")

def get_args():
    parser.add_argument('-tr', '--trainpath', type=str, help='train image datapath')
    parser.add_argument('-te', '--testpath', type=str, help='test image datapath')
    parser.add_argument('-l', '--learningRate', type=float, default=0.0002,help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=30,help='Number of epochs')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    unet_model = UNet()
    vgg_feature_extractor = VGGFeatureExtractor()

    unet_model.to(device)
    vgg_feature_extractor.to(device)
    pixelwise_loss = PixelwiseLoss().to(device)
    
    learning_rate = args.learningRate
    train_path = args.trainpath
    test_path = args.testpath
    epochs = args.epochs
    optimizer = optim.Adam(unet_model.parameters(), lr=learning_rate,nesterov=True)
    
    train_test_model(model, train_path, test_path, feature_loss, pixelwise_loss, optimizer, device, num_epochs=epochs)
