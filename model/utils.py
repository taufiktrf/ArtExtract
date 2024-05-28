import matplotlib.pyplot as plt
import numpy as np

# Display the output image and compare with target image visually
def plot_images(output_image, target_image, epoch, channel):
    output_image_np = output_image.squeeze().cpu().detach().numpy()
    target_image_np = target_image.squeeze().cpu().detach().numpy()
    plt.figure(figsize=(7, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(output_image_np, cmap='gray')
    plt.title(f'Output - Epoch {epoch} - Channel {channel}',fontsize=9)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(target_image_np, cmap='gray')
    plt.title(f'Target - Epoch {epoch} - Channel {channel}',fontsize=9)
    plt.axis('off')
    
    plt.show()
    
def viz_train(train_losses, val_losses, train_rrmses, val_rrmses):
    # Plotting Losses
    train_losses = [loss.cpu().detach().numpy() for loss in train_losses]
    val_losses = [loss.cpu().detach().numpy() for loss in val_losses]
    
    train_rrmses = [rrmse.cpu().detach().numpy() for rrmse in train_rrmses]
    val_rrmses = [rrmse.cpu().detach().numpy() for rrmse in val_rrmses]
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(np.arange(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

    # Plotting PSNRs
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(train_rrmses) + 1), train_rrmses, label='Train rRMSE')
    plt.plot(np.arange(1, len(val_rrmses) + 1), val_rrmses, label='Val rRMSE')
    plt.xlabel('Epoch')
    plt.ylabel('rRMSE')
    plt.title('Training and Validation rRMSE')
    plt.legend()
    plt.show()