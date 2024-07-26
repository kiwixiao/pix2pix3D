import argparse
import torch
import SimpleITK as sitk
import numpy as np
from model import Generator
from utils import resample_image, preprocess_data

def load_checkpoint(checkpoint_path, device):
    """Load the trained generator from a checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator = Generator(in_channels=1, out_channels=1, features=64).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    return generator

def predict(generator, input_image, device):
    """Generate a prediction mask for the input image"""
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0).float().to(device)
        output = generator(input_tensor)
    return output.squeeze().cpu().numpy()

def main(args):
    # Set up device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained generator
    generator = load_checkpoint(args.checkpoint, device)

    # Load and preprocess input image
    original_image = sitk.ReadImage(args.input_image)
    input_image, _ = preprocess_data(args.input_image, args.input_image, args.target_shape)

    # Generate prediction
    predicted_mask = predict(generator, input_image, device)

    # Convert predicted mask back to original image space
    predicted_mask_sitk = sitk.GetImageFromArray(predicted_mask)
    predicted_mask_sitk.SetSpacing(original_image.GetSpacing())
    predicted_mask_sitk.SetOrigin(original_image.GetOrigin())
    predicted_mask_sitk.SetDirection(original_image.GetDirection())

    # Resample the predicted mask to match the original image
    resampled_mask = sitk.Resample(predicted_mask_sitk, original_image, sitk.Transform(), sitk.sitkNearestNeighbor)

    # Save the predicted mask
    output_path = args.input_image.replace('.nii.gz', '_predicted_mask.nii.gz')
    sitk.WriteImage(resampled_mask, output_path)
    print(f"Predicted mask saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for 3D pix2pixGAN MRI segmentation")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("input_image", type=str, help="Path to the input MRI image file (.nii.gz)")
    parser.add_argument("--target_shape", nargs=3, type=int, default=[128, 128, 128], help="Target shape for resampling")
    args = parser.parse_args()

    main(args)