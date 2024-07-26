import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from utils import plot_prediction

def train(generator, discriminator, train_mri, train_masks, test_data, args, device):
    logger = args.logger

    # Define loss functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.L1Loss()

    # Set up optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Create dataset and dataloader
    train_dataset = TensorDataset(torch.from_numpy(train_mri).float(), torch.from_numpy(train_masks).float())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Prepare test data
    test_mri, test_mask = test_data
    test_mri_tensor = torch.from_numpy(test_mri).unsqueeze(0).float().to(device)

    for epoch in range(args.num_epochs):
        for i, (mri, true_mask) in enumerate(train_loader):
            mri, true_mask = mri.to(device), true_mask.to(device)
            
            # Log tensor sizes for debugging
            logger.info(f"MRI tensor size: {mri.size()}, True mask tensor size: {true_mask.size()}")

            # Train Discriminator
            optimizer_d.zero_grad()
            
            # Generate fake mask
            fake_mask = generator(mri)
            
            # Train on fake samples
            pred_fake = discriminator(torch.cat([mri, fake_mask], dim=1))
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            
            # Train on real samples
            pred_real = discriminator(torch.cat([mri, true_mask], dim=1))
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            
            # Total discriminator loss
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            
            # Generate fake mask
            fake_mask = generator(mri)
            pred_fake = discriminator(torch.cat([mri, fake_mask], dim=1))
            
            # Adversarial loss
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            # Pixel-wise loss
            loss_g_pixel = criterion_pixel(fake_mask, true_mask)
            
            # Total generator loss
            loss_g = loss_g_gan + 100 * loss_g_pixel
            loss_g.backward()
            optimizer