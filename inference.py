import argparse
import torch
import SimpleITK as sitk
import numpy as np
from model import Generator
from utils import preprocess_data, resize_volume, extract_patches
import os

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator = Generator(in_channels=1, out_channels=1, features=64).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    return generator

def predict_patch(generator, patch, device):
    with torch.no_grad():
        input_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
        output = generator(input_tensor)
    return output.squeeze().cpu().numpy()

def reconstruct_from_patches(patches, original_shape, patch_size, stride):
    output = np.zeros(original_shape)
    count = np.zeros(original_shape)
    
    i = 0
    for z in range(0, original_shape[0] - patch_size[0] + 1, stride[0]):
        for y in range(0, original_shape[1] - patch_size[1] + 1, stride[1]):
            for x in range(0, original_shape[2] - patch_size[2] + 1, stride[2]):
                output[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += patches[i]
                count[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += 1
                i += 1
    
    output = np.divide(output, count, where=count!=0)
    return output

def predict(generator, input_image, device, patch_size, stride):
    patches, _ = extract_patches(input_image, None, patch_size, stride)
    predicted_patches = []
    
    for patch in patches:
        pred_patch = predict_patch(generator, patch, device)
        predicted_patches.append(pred_patch)
    
    predicted_mask = reconstruct_from_patches(predicted_patches, input_image.shape, patch_size, stride)
    return predicted_mask

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = load_checkpoint(args.checkpoint, device)

    # Load and preprocess input image
    original_image = sitk.ReadImage(args.input_image)
    preprocessed_image, _, _ = preprocess_data(args.input_image, None, args.target_shape, args.new_spacing)

    # Predict on preprocessed image
    predicted_mask = predict(generator, preprocessed_image, device, args.patch_size, args.stride)

    # Transform prediction from [-1, 1] to [0, 1]
    predicted_mask = (predicted_mask + 1) / 2

    # Threshold the prediction
    predicted_mask = (predicted_mask > 0.5).astype(np.float32)

    # Resize prediction to original image size
    original_size = original_image.GetSize()[::-1]  # SimpleITK uses (x,y,z) ordering
    predicted_mask_resized = resize_volume(predicted_mask, original_size)

    # Save the predicted mask
    output_image = sitk.GetImageFromArray(predicted_mask_resized)
    output_image.CopyInformation(original_image)
    output_path = os.path.splitext(args.input_image)[0] + '_predicted_mask.nii.gz'
    sitk.WriteImage(output_image, output_path)
    print(f"Predicted mask saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for 3D pix2pixGAN MRI segmentation")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("input_image", type=str, help="Path to the input MRI image file (.nii.gz)")
    parser.add_argument("--target_shape", nargs=3, type=int, default=[128, 128, 128], help="Target shape for resampling")
    parser.add_argument("--new_spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0], help="New spacing for resampling")
    parser.add_argument("--patch_size", nargs=3, type=int, default=[64, 64, 64], help="Size of patches for prediction")
    parser.add_argument("--stride", nargs=3, type=int, default=[32, 32, 32], help="Stride for patch extraction")
    args = parser.parse_args()

    main(args)