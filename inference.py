import argparse
import torch
import SimpleITK as sitk
import numpy as np
from model import Generator
from utils import resample_image, preprocess_data, check_tensor_size

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator = Generator(in_channels=1, out_channels=1, features=64).to(device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    return generator

def predict_patch(generator, patch, device, expected_input_size):
    with torch.no_grad():
        input_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
        check_tensor_size(input_tensor, expected_input_size, "Input patch")
        output = generator(input_tensor)
        check_tensor_size(output, expected_input_size, "Output patch")
    return output.squeeze().cpu().numpy()

def create_weight_mask(patch_size):
    """Create a weight mask for linear blending"""
    weight_mask = np.ones(patch_size)
    for axis in range(3):
        grad = np.linspace(0, 1, patch_size[axis])
        sl = [np.newaxis] * 3
        sl[axis] = slice(None)
        weight_mask *= np.minimum(grad[tuple(sl)], grad[tuple(sl)][::-1])
    return weight_mask

def predict(generator, input_image, device, patch_size, stride):
    """Predict using sliding window with overlap and blending"""
    prediction = np.zeros_like(input_image)
    weight_sum = np.zeros_like(input_image)
    weight_mask = create_weight_mask(patch_size)

    expected_input_size = torch.Size([1, 1] + list(patch_size))

    for z in range(0, input_image.shape[0] - patch_size[0] + 1, stride[0]):
        for y in range(0, input_image.shape[1] - patch_size[1] + 1, stride[1]):
            for x in range(0, input_image.shape[2] - patch_size[2] + 1, stride[2]):
                patch = input_image[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                pred_patch = predict_patch(generator, patch, device, expected_input_size)
                
                prediction[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += pred_patch * weight_mask
                weight_sum[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]] += weight_mask

    # Normalize by weight sum to get final prediction
    prediction = np.divide(prediction, weight_sum, where=weight_sum != 0)
    return prediction

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = load_checkpoint(args.checkpoint, device)

    original_image = sitk.ReadImage(args.input_image)
    input_image, _ = preprocess_data(args.input_image, args.input_image, args.target_shape)

    predicted_mask = predict(generator, input_image, device, args.patch_size, args.stride)

    # Convert prediction to binary mask
    predicted_mask = (predicted_mask > 0.5).astype(np.float32)

    predicted_mask_sitk = sitk.GetImageFromArray(predicted_mask)
    predicted_mask_sitk.SetSpacing(original_image.GetSpacing())
    predicted_mask_sitk.SetOrigin(original_image.GetOrigin())
    predicted_mask_sitk.SetDirection(original_image.GetDirection())

    resampled_mask = sitk.Resample(predicted_mask_sitk, original_image, sitk.Transform(), sitk.sitkNearestNeighbor)

    output_path = args.input_image.replace('.nii.gz', '_predicted_mask.nii.gz')
    sitk.WriteImage(resampled_mask, output_path)
    print(f"Predicted mask saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for 3D pix2pixGAN MRI segmentation")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("input_image", type=str, help="Path to the input MRI image file (.nii.gz)")
    parser.add_argument("--target_shape", nargs=3, type=int, default=[128, 128, 128], help="Target shape for resampling")
    parser.add_argument("--patch_size", nargs=3, type=int, default=[64, 64, 64], help="Size of patches for prediction")
    parser.add_argument("--stride", nargs=3, type=int, default=[32, 32, 32], help="Stride for patch extraction")
    args = parser.parse_args()

    main(args)