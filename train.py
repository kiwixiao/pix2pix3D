import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Generator, Discriminator, ComboLoss, FocalTverskyLoss
from utils import plot_prediction, calculate_metrics
from torch.utils.tensorboard import SummaryWriter

def train(generator, discriminator, train_dataset, test_dataset, args):
    device = args.device
    logger = args.logger
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs'))

    # Loss functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_seg = ComboLoss()
    criterion_focal_tversky = FocalTverskyLoss()

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Lists to store metrics for plotting
    epochs = []
    train_losses_d = []
    train_losses_g = []
    train_losses_g_gan = []
    train_losses_g_seg = []
    train_losses_g_focal = []
    val_dice_scores = []
    val_iou_scores = []

    for epoch in range(args.num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_loss_d = 0
        epoch_loss_g = 0
        epoch_loss_g_gan = 0
        epoch_loss_g_seg = 0
        epoch_loss_g_focal = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for i, (mri_patch, mask_patch) in enumerate(pbar):
            mri_patch, mask_patch = mri_patch.to(device), mask_patch.to(device)
            batch_size = mri_patch.size(0)

            # Train Discriminator
            optimizer_d.zero_grad()
            
            fake_patch = generator(mri_patch)
            real_input = torch.cat([mri_patch, mask_patch], dim=1)
            fake_input = torch.cat([mri_patch, fake_patch.detach()], dim=1)
            
            pred_real = discriminator(real_input)
            pred_fake = discriminator(fake_input)
            
            loss_d_real = criterion_gan(pred_real, torch.ones(batch_size, 1, 1, 1, 1).to(device))
            loss_d_fake = criterion_gan(pred_fake, torch.zeros(batch_size, 1, 1, 1, 1).to(device))
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            
            loss_d.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            
            fake_patch = generator(mri_patch)
            fake_input = torch.cat([mri_patch, fake_patch], dim=1)
            pred_fake = discriminator(fake_input)
            
            loss_g_gan = criterion_gan(pred_fake, torch.ones(batch_size, 1, 1, 1, 1).to(device))
            loss_g_seg = criterion_seg(fake_patch, mask_patch)
            loss_g_focal = criterion_focal_tversky(fake_patch, mask_patch)
            
            loss_g = loss_g_gan + args.lambda_seg * loss_g_seg + args.lambda_focal * loss_g_focal
            
            loss_g.backward()
            optimizer_g.step()

            # Accumulate losses
            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            epoch_loss_g_gan += loss_g_gan.item()
            epoch_loss_g_seg += loss_g_seg.item()
            epoch_loss_g_focal += loss_g_focal.item()

            pbar.set_postfix({'D_loss': loss_d.item(), 'G_loss': loss_g.item()})

        # Calculate average losses for the epoch
        avg_loss_d = epoch_loss_d / len(train_loader)
        avg_loss_g = epoch_loss_g / len(train_loader)
        avg_loss_g_gan = epoch_loss_g_gan / len(train_loader)
        avg_loss_g_seg = epoch_loss_g_seg / len(train_loader)
        avg_loss_g_focal = epoch_loss_g_focal / len(train_loader)

        logger.info(f"Epoch [{epoch+1}/{args.num_epochs}], D Loss: {avg_loss_d:.4f}, G Loss: {avg_loss_g:.4f}")

        # Validation
        generator.eval()
        val_dice = 0
        val_iou = 0
        num_val_samples = 0

        with torch.no_grad():
            for mri_patch, mask_patch in test_loader:
                mri_patch, mask_patch = mri_patch.to(device), mask_patch.to(device)
                pred_mask = generator(mri_patch)
                pred_mask = (pred_mask > 0).float()
                metrics = calculate_metrics(mask_patch.cpu().numpy(), pred_mask.cpu().numpy())
                val_dice += metrics['dice']
                val_iou += metrics['iou']
                num_val_samples += 1

        avg_val_dice = val_dice / num_val_samples
        avg_val_iou = val_iou / num_val_samples

        logger.info(f"Validation - Dice: {avg_val_dice:.4f}, IoU: {avg_val_iou:.4f}")

        # Log to TensorBoard
        writer.add_scalar('Loss/Discriminator', avg_loss_d, epoch)
        writer.add_scalar('Loss/Generator', avg_loss_g, epoch)
        writer.add_scalar('Loss/Generator_GAN', avg_loss_g_gan, epoch)
        writer.add_scalar('Loss/Generator_Segmentation', avg_loss_g_seg, epoch)
        writer.add_scalar('Loss/Generator_FocalTversky', avg_loss_g_focal, epoch)
        writer.add_scalar('Metrics/Dice', avg_val_dice, epoch)
        writer.add_scalar('Metrics/IoU', avg_val_iou, epoch)

        # Save model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # Plot and save sample prediction
        if (epoch + 1) % args.plot_interval == 0:
            sample_mri, sample_mask = next(iter(test_loader))
            sample_mri, sample_mask = sample_mri.to(device), sample_mask.to(device)
            with torch.no_grad():
                sample_pred = generator(sample_mri)
            plot_prediction(sample_mri.squeeze().cpu().numpy(), 
                            sample_mask.squeeze().cpu().numpy(), 
                            sample_pred.squeeze().cpu().numpy(), 
                            epoch, args.output_dir)

        # Append metrics for final plotting
        epochs.append(epoch + 1)
        train_losses_d.append(avg_loss_d)
        train_losses_g.append(avg_loss_g)
        train_losses_g_gan.append(avg_loss_g_gan)
        train_losses_g_seg.append(avg_loss_g_seg)
        train_losses_g_focal.append(avg_loss_g_focal)
        val_dice_scores.append(avg_val_dice)
        val_iou_scores.append(avg_val_iou)

    writer.close()

    # Plot and save training curves
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_losses_d, label='Discriminator Loss')
    plt.plot(epochs, train_losses_g, label='Generator Loss')
    plt.plot(epochs, train_losses_g_gan, label='Generator GAN Loss')
    plt.plot(epochs, train_losses_g_seg, label='Generator Segmentation Loss')
    plt.plot(epochs, train_losses_g_focal, label='Generator Focal Tversky Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'training_losses.png'))
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, val_dice_scores, label='Dice Score')
    plt.plot(epochs, val_iou_scores, label='IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'validation_metrics.png'))
    plt.close()

    logger.info("Training completed.")