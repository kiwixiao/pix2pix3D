import argparse
from datetime import datetime
import os
import torch
from utils import setup_logger, load_dataset
from model import Generator, Discriminator
from train import train

def main(args):
    # Set up logging
    logger = setup_logger()
    
    # Set up device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    mri_images, mask_labels = load_dataset(args.data_dir, args.target_shape)
    
    # Split data into train and test sets (using last image as test)
    test_idx = len(mri_images) - 1
    train_mri = mri_images[:test_idx]
    train_masks = mask_labels[:test_idx]
    test_mri = mri_images[test_idx]
    test_mask = mask_labels[test_idx]

    # Initialize generator and discriminator models
    generator = Generator(args.in_channels, args.out_channels, args.features).to(device)
    discriminator = Discriminator(args.in_channels + args.out_channels, args.features).to(device)

    # Create output directory for saving results
    os.makedirs(args.output_dir, exist_ok=True)

    # Train the model
    train(generator, discriminator, train_mri, train_masks, (test_mri, test_mask), args, device)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="3D pix2pixGAN for MRI segmentation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for saving results")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--out_channels", type=int, default=1, help="Number of output channels")
    parser.add_argument("--features", type=int, default=64, help="Number of features in the first layer")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval for saving checkpoints")
    parser.add_argument("--plot_interval", type=int, default=5, help="Interval for plotting predictions")
    parser.add_argument("--target_shape", nargs=3, type=int, default=[128, 128, 128], help="Target shape for resampling")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")

    args = parser.parse_args()
    
    # Create a unique output directory based on current timestamp
    args.output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    main(args)