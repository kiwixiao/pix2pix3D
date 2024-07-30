import argparse
import torch
import SimpleITK as sitk
import numpy as np
from model import Generator
from utils import preprocess_data, resize_volume
from tqdm import tqdm
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

def create_linear_ramp(size):
    return np.linspace(0, 1, size)

def predict(generator, input_image, device, patch_size, stride):
    d, h, w = input_image.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    # Initialize the output and weight arrays
    output = np.zeros_like(input_image, dtype=float)
    weight = np.zeros_like(input_image, dtype=float)

    # Create linear ramps for blending
    ramp_d = create_linear_ramp(pd)
    ramp_h = create_linear_ramp(ph)
    ramp_w = create_linear_ramp(pw)

    # Create the 3D linear blend kernel
    kernel = np.ones(patch_size, dtype=float)
    kernel = np.minimum(kernel, ramp_d[:, np.newaxis, np.newaxis])
    kernel = np.minimum(kernel, ramp_h[np.newaxis, :, np.newaxis])
    kernel = np.minimum(kernel, ramp_w[np.newaxis, np.newaxis, :])
    kernel = np.minimum(kernel, np.flip(kernel, axis=(0,1,2)))

    # Sliding window prediction
    for z in tqdm(range(0, d - pd + 1, sd), desc="Predicting"):
        for y in range(0, h - ph + 1, sh):
            for x in range(0, w - pw + 1, sw):
                patch = input_image[z:z+pd, y:y+ph, x:x+pw]
                pred_patch = predict_patch(generator, patch, device)
                
                output[z:z+pd, y:y+ph, x:x+pw] += pred_patch * kernel
                weight[z:z+pd, y:y+ph, x:x+pw] += kernel

    # Normalize the output by the accumulated weights
    output = np.divide(output, weight, where=weight!=0)
    
    return output

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = load_checkpoint(args.checkpoint, device)

    # Load and preprocess input image
    original_image = sitk.ReadImage(args.input_image)
    preprocessed_image, _, _ = preprocess_data(args.input_image, None, args.target_shape, args.new_spacing)

    # Predict on preprocessed image
    predicted_mask = predict(generator, preprocessed_image, device, args.patch_size, args.stride)

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
    parser.add_argument("--target_shape", nargs=3, type=int, default=[256, 256, 256], help="Target shape for resampling")
    parser.add_argument("--new_spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0], help="New spacing for resampling")
    parser.add_argument("--patch_size", nargs=3, type=int, default=[32, 32, 32], help="Size of patches for prediction")
    parser.add_argument("--stride", nargs=3, type=int, default=[16, 16, 16], help="Stride for patch extraction")
    args = parser.parse_args()

    main(args)