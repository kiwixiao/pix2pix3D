import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, jaccard_score
from utils import plot_prediction

def calculate_dice(true_mask, pred_mask, smooth=1e-7):
    """Calculate Dice coefficient."""
    intersection = np.sum(true_mask * pred_mask)
    return (2. * intersection + smooth) / (np.sum(true_mask) + np.sum(pred_mask) + smooth)

def calculate_metrics(true_mask, pred_mask, threshold=0.5):
    """Calculate various segmentation metrics."""
    true_mask = true_mask.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()
    pred_mask_binary = (pred_mask > threshold).astype(np.float32)
    
    
    dice = calculate_dice(true_mask, pred_mask_binary)
    true_flat = true_mask.flatten()
    pred_flat = pred_mask_binary.flatten()
    
    print(f"True mask shape: {true_mask.shape}, unique values: {np.unique(true_mask)}")
    print(f"Pred mask shape: {pred_mask_binary.shape}, unique values: {np.unique(pred_mask_binary)}")
    
    f1 = f1_score(true_flat, pred_flat, average='binary',zero_division=1)
    iou = jaccard_score(true_flat, pred_flat, average='binary', zero_division=1)
    
    return {'dice': dice, 'f1': f1, 'iou': iou}

def validate(generator, test_loader, device, args):
    """Perform validation and calculate metrics."""
    generator.eval()
    val_metrics = {'dice': 0, 'f1': 0, 'iou': 0}
    num_samples = 0

    with torch.no_grad():
        for mri_patch, true_mask in test_loader:
            mri_patch, true_mask = mri_patch.to(device), true_mask.to(device)
            pred_mask = generator(mri_patch)
            metrics = calculate_metrics(true_mask, pred_mask)
            
            for key in val_metrics:
                val_metrics[key] += metrics[key]
            num_samples += 1

    for key in val_metrics:
        val_metrics[key] /= num_samples

    return val_metrics

def train(generator, discriminator, train_dataset, test_dataset, args):
    device = args.device
    logger = args.logger

    # Set up TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs'))

    # Loss functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_seg = nn.L1Loss()  # You might want to use a different segmentation loss

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    best_dice = 0.0
    
    # Lists to store metrics for plotting
    train_losses_d = []
    train_losses_g = []
    val_dice_scores = []
    val_f1_scores = []
    val_iou_scores = []

    for epoch in range(args.num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_loss_d = 0
        epoch_loss_g = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for i, (mri_patch, mask_patch) in enumerate(pbar):
            mri_patch, mask_patch = mri_patch.to(device), mask_patch.to(device)
            batch_size = mri_patch.size(0)

            # Train Discriminator
            optimizer_d.zero_grad()
            
            fake_patch = generator(mri_patch)
            print(f"Generator output shape: {fake_patch.shape}, range: [{fake_patch.min().item()}, {fake_patch.max().item()}]")
            real_input = torch.cat([mri_patch, mask_patch], dim=1)
            fake_input = torch.cat([mri_patch, fake_patch.detach()], dim=1)
            
            pred_real = discriminator(real_input)
            pred_fake = discriminator(fake_input)
            
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            
            loss_d.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            
            fake_patch = generator(mri_patch)
            fake_input = torch.cat([mri_patch, fake_patch], dim=1)
            pred_fake = discriminator(fake_input)
            
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_g_seg = criterion_seg(fake_patch, mask_patch)
            
            loss_g = loss_g_gan + args.lambda_seg * loss_g_seg
            
            loss_g.backward()
            optimizer_g.step()

            # Update progress bar
            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            pbar.set_postfix({'D_loss': loss_d.item(), 'G_loss': loss_g.item()})

            # Log to TensorBoard (every N steps)
            if i % args.log_interval == 0:
                step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/Discriminator', loss_d.item(), step)
                writer.add_scalar('Loss/Generator', loss_g.item(), step)
                writer.add_scalar('Loss/Generator_GAN', loss_g_gan.item(), step)
                writer.add_scalar('Loss/Generator_Segmentation', loss_g_seg.item(), step)

        # Calculate average losses for the epoch
        avg_loss_d = epoch_loss_d / len(train_loader)
        avg_loss_g = epoch_loss_g / len(train_loader)
        train_losses_d.append(avg_loss_d)
        train_losses_g.append(avg_loss_g)

        logger.info(f"Epoch [{epoch+1}/{args.num_epochs}], D Loss: {avg_loss_d:.4f}, G Loss: {avg_loss_g:.4f}")

        # Validation
        val_metrics = validate(generator, test_loader, device, args)
        val_dice_scores.append(val_metrics['dice'])
        val_f1_scores.append(val_metrics['f1'])
        val_iou_scores.append(val_metrics['iou'])
        logger.info(f"Validation metrics: Dice={val_metrics['dice']:.4f}, F1={val_metrics['f1']:.4f}, IoU={val_metrics['iou']:.4f}")

        # Log validation metrics to TensorBoard
        writer.add_scalar('Validation/Dice', val_metrics['dice'], epoch)
        writer.add_scalar('Validation/F1', val_metrics['f1'], epoch)
        writer.add_scalar('Validation/IoU', val_metrics['iou'], epoch)

        # Save model every N epochs
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
            }, os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'))
            logger.info(f"Model saved at epoch {epoch+1}")

        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f"New best model saved with Dice score: {best_dice:.4f}")

        # Plot and save sample prediction
        if (epoch + 1) % args.plot_interval == 0:
            generator.eval()
            with torch.no_grad():
                sample_mri, sample_mask = next(iter(test_loader))
                sample_mri, sample_mask = sample_mri.to(device), sample_mask.to(device)
                sample_pred = generator(sample_mri)
                plot_prediction(sample_mri.squeeze().cpu().numpy(), 
                                sample_mask.squeeze().cpu().numpy(), 
                                sample_pred.squeeze().cpu().numpy(), 
                                epoch, args.output_dir)
                
                # Log images to TensorBoard
                writer.add_image('Sample/MRI', sample_mri.squeeze(), epoch, dataformats='CHW')
                writer.add_image('Sample/True_Mask', sample_mask.squeeze(), epoch, dataformats='CHW')
                writer.add_image('Sample/Predicted_Mask', sample_pred.squeeze(), epoch, dataformats='CHW')
            
            generator.train()

    # Close TensorBoard writer
    writer.close()

    # Plot and save training curves
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, args.num_epochs+1), train_losses_d, label='Discriminator Loss')
    plt.plot(range(1, args.num_epochs+1), train_losses_g, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'training_losses.png'))
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, args.num_epochs+1), val_dice_scores, label='Dice Score')
    plt.plot(range(1, args.num_epochs+1), val_f1_scores, label='F1 Score')
    plt.plot(range(1, args.num_epochs+1), val_iou_scores, label='IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'validation_metrics.png'))
    plt.close()

    logger.info("Training completed.")