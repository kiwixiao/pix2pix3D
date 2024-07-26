import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from utils import plot_prediction, check_tensor_size
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter

def train(generator, discriminator, patch_dataset, test_data, args, device):
    logger = args.logger

    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs'))

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.L1Loss()

    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    train_loader = DataLoader(patch_dataset, batch_size=args.batch_size, shuffle=True)

    test_mri, test_mask = test_data
    test_mri_tensor = torch.from_numpy(test_mri).unsqueeze(0).unsqueeze(0).float().to(device)

    expected_input_size = torch.Size([args.batch_size, 1] + list(args.patch_size))
    expected_output_size = torch.Size([args.batch_size, 1] + list(args.patch_size))

    # Save model architecture as PNG
    dummy_input = torch.randn(1, 1, *args.patch_size).to(device)
    y = generator(dummy_input)
    make_dot(y, params=dict(generator.named_parameters())).render(os.path.join(args.output_dir, "generator_architecture"), format="png")
    
    y = discriminator(torch.cat([dummy_input, y], dim=1))
    make_dot(y, params=dict(discriminator.named_parameters())).render(os.path.join(args.output_dir, "discriminator_architecture"), format="png")

    # Log model graphs to TensorBoard
    writer.add_graph(generator, dummy_input)
    writer.add_graph(discriminator, torch.cat([dummy_input, y], dim=1))

    global_step = 0
    for epoch in range(args.num_epochs):
        generator.train()
        discriminator.train()
        
        for i, (mri_patch, mask_patch) in enumerate(train_loader):
            mri_patch, mask_patch = mri_patch.unsqueeze(1).float().to(device), mask_patch.unsqueeze(1).float().to(device)
            
            # Check tensor sizes
            check_tensor_size(mri_patch, expected_input_size, "MRI patch")
            check_tensor_size(mask_patch, expected_output_size, "Mask patch")

            # Train Discriminator
            optimizer_d.zero_grad()
            
            fake_patch = generator(mri_patch)
            check_tensor_size(fake_patch, expected_output_size, "Generated patch")

            pred_fake = discriminator(torch.cat([mri_patch, fake_patch.detach()], dim=1))
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            
            pred_real = discriminator(torch.cat([mri_patch, mask_patch], dim=1))
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            
            fake_patch = generator(mri_patch)
            check_tensor_size(fake_patch, expected_output_size, "Generated patch")

            pred_fake = discriminator(torch.cat([mri_patch, fake_patch], dim=1))
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_g_pixel = criterion_pixel(fake_patch, mask_patch)
            
            loss_g = loss_g_gan + 100 * loss_g_pixel
            loss_g.backward()
            optimizer_g.step()

            # Log losses to TensorBoard
            writer.add_scalar('Loss/Discriminator', loss_d.item(), global_step)
            writer.add_scalar('Loss/Generator', loss_g.item(), global_step)
            writer.add_scalar('Loss/Generator_GAN', loss_g_gan.item(), global_step)
            writer.add_scalar('Loss/Generator_Pixel', loss_g_pixel.item(), global_step)

            global_step += 1

        logger.info(f"Epoch [{epoch+1}/{args.num_epochs}], D Loss: {loss_d.item():.4f}, G Loss: {loss_g.item():.4f}")

        # Save model after every epoch
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
        }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        logger.info(f"Checkpoint saved at epoch {epoch+1}")

        if (epoch + 1) % args.plot_interval == 0:
            generator.eval()
            with torch.no_grad():
                pred_mask = generator(test_mri_tensor)
                plot_prediction(test_mri, test_mask, pred_mask.squeeze().cpu().numpy(), epoch, args.output_dir)
                logger.info(f"Prediction plot saved at epoch {epoch+1}")
            
            # Log images to TensorBoard
            writer.add_images('MRI/Test', test_mri_tensor, epoch)
            writer.add_images('Mask/True', torch.from_numpy(test_mask).unsqueeze(0).unsqueeze(0), epoch)
            writer.add_images('Mask/Predicted', pred_mask, epoch)

    writer.close()