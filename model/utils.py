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
    
def viz_train(losses, train_rrmses, val_rrmses,train_psnrs,val_psnrs,train_ssims,val_ssims):
    # Plotting Losses
    train_losses = [loss.cpu().detach().numpy() for loss in losses]    
    train_rrmses = [rrmse.cpu().detach().numpy() for rrmse in train_rrmses]
    train_psnrs = [psnr.cpu().detach().numpy() for psnr in train_psnrs]
    train_ssims = [ssim.cpu().detach().numpy() for ssim in train_ssims]
    
    val_rrmses = [rrmse.cpu().detach().numpy() for rrmse in val_rrmses]
    val_psnrs = [psnr.cpu().detach().numpy() for psnr in val_psnrs]
    val_ssims = [ssim.cpu().detach().numpy() for ssim in val_ssims]
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(train_losses) + 1), train_losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
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
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(train_psnrs) + 1), train_psnrs, label='Train PSNR')
    plt.plot(np.arange(1, len(val_psnrs) + 1), val_psnrs, label='Val PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Training and Validation PSNR')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(train_ssims) + 1), train_ssims, label='Train SSIM')
    plt.plot(np.arange(1, len(val_ssims) + 1), val_ssims, label='Val SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Training and Validation SSIM')
    plt.legend()
    plt.show()