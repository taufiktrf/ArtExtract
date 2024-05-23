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
    
def viz_train(train_losses, val_losses, train_psnrs, val_psnrs):
    # Plotting Losses
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
    plt.plot(np.arange(1, len(train_psnrs) + 1), train_psnrs, label='Train PSNR')
    plt.plot(np.arange(1, len(val_psnrs) + 1), val_psnrs, label='Val PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Training and Validation PSNRs')
    plt.legend()
    plt.show()