import logging
import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
from sklearn.metrics import f1_score, jaccard_score


def setup_logger():
    """Set up and return a logger"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def extract_roi(image, mask=None, margin=20):
    if mask is None:
        # if no mask is provided, use the entire image
        return image, None
    z, y, x = np.where(mask > 0)
    z_min, z_max = max(z_min() - margin, 0), min(z.max() + margin, mask.shape[0])
    y_min, y_max = max(y_min() - margin, 0), min(y.max() + margin, mask.shape[1])
    x_min, x_max = max(x_min() - margin, 0), min(x.max() + margin, mask.shape[2])
    
    roi_image = image[z_min:z_max, y_min:y_max, x_min:x_max]
    roi_mask = mask[z_min:z_max, y_min:y_max, x_min:x_max] if mask is not None else None

    return roi_image, roi_mask, (z_min, z_max, y_min, y_max, x_min, x_max)

def preprocess_data(image_path, mask_path, target_shape):
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path) if mask_path else None

    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask) if mask else None
    
    roi_image, roi_mask, roi_coords = extract_roi(image_array, mask_array)

    # normalized roi image
    

def resample_image(image, new_spacing=[1.0, 1.0, 1.0]):
    """Resample the image to a new spacing"""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Calculate new size based on the new spacing
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]
    
    # Set up the resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    
    return resample.Execute(image)

def preprocess_data(image_path, mask_path, target_shape):
    """Preprocess the image and mask data"""
    # Read the image and mask
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    
    # Resample to isotropic resolution
    image_resampled = resample_image(image)
    mask_resampled = resample_image(mask)
    
    # Convert to numpy arrays
    image_array = sitk.GetArrayFromImage(image_resampled)
    mask_array = sitk.GetArrayFromImage(mask_resampled)
    
    # Normalize image intensity
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min())
    
    # Resize to target shape
    image_resized = sitk.GetImageFromArray(image_array)
    mask_resized = sitk.GetImageFromArray(mask_array)
    
    image_resized = sitk.Resample(image_resized, target_shape, sitk.Transform(), sitk.sitkLinear)
    mask_resized = sitk.Resample(mask_resized, target_shape, sitk.Transform(), sitk.sitkNearestNeighbor)
    
    image_final = sitk.GetArrayFromImage(image_resized)
    mask_final = sitk.GetArrayFromImage(mask_resized)
    
    return image_final, mask_final

def load_dataset(data_dir, target_shape):
    """Load and preprocess all images in the data directory"""
    mri_images = []
    mask_labels = []
    
    # Get all MRI and mask files
    mri_files = sorted(glob.glob(os.path.join(data_dir, '*_mri.nii*')))
    mask_files = sorted(glob.glob(os.path.join(data_dir, '*_mask.nii*')))
    
    # Process each pair of MRI and mask
    for mri_file, mask_file in zip(mri_files, mask_files):
        image, mask = preprocess_data(mri_file, mask_file, target_shape)
        mri_images.append(image)
        mask_labels.append(mask)
    
    return np.array(mri_images), np.array(mask_labels)

def extract_patches(image, patch_size, stride):
    # extract patches from a 3D image
    patches = []
    for z in range(0, image.shape[0] - patch_size[0] + 1, stride[0]):
        for y in range(0, image.shape[1] - patch_size[1] + 1, stride[1]):
            for x in range(0, image.shape[2] - patch_size[2] + 1, stride[2]):
                patch = image[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                patches.append(patch)
    return np.array(patches)

class PatchDataset(Dataset):
    def __init__(self, mri_images, mask_labels, patch_size=(64,64,64), stride=(32,32,32)):
        self.mri_images=mri_images
        self.mask_labels = mask_labels
        self.patch_size = patch_size
        self.stride = stride

        self.mri_patches = []
        self.mask_patches = []

        for mri, mask in zip(mri_images, mask_labels):
            mri_patches = extract_patches(mri, patch_size, stride)
            mask_patches = extract_patches(mask, patch_size, stride)

            self.mri_patches.extend(mri_patches)
            self.mask_patches.extend(mask_patches)

        self.mri_patches = np.array(self.mri_patches)
        self.mask_patches = np.array(self.mask_patches)

    def __len__(self):
        return len(self.mri_patches)

    def __getitem__(self, idx):
        mri_patch = self.mri_patches[idx]
        mask_patch = self.mask_patches[idx]
        return mri_patch, mask_patch

                

def plot_prediction(test_mri, test_mask, pred_mask, epoch, output_dir):
    """Plot and save the prediction results"""
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i, dim in enumerate(['Sagittal', 'Coronal', 'Axial']):
        slice_idx = test_mri.shape[i] // 2
        
        # Plot original MRI
        axs[i, 0].imshow(np.take(test_mri, slice_idx, axis=i), cmap='gray')
        axs[i, 0].set_title(f'{dim} MRI')
        
        # Plot true mask
        axs[i, 1].imshow(np.take(test_mask, slice_idx, axis=i), cmap='gray')
        axs[i, 1].set_title(f'{dim} True Mask')
        
        # Plot predicted mask
        axs[i, 2].imshow(np.take(pred_mask, slice_idx, axis=i), cmap='gray')
        axs[i, 2].set_title(f'{dim} Predicted Mask')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'prediction_epoch_{epoch+1}.png'))
    plt.close()

def check_tensor_size(tensor, expected_size, name=""):
    """
    Check if the tensor size matches the expected size.
    If not, print an error message and raise an exception.
    """
    if tensor.size() != expected_size:
        error_msg = f"Size mismatch for {name}. Expected {expected_size}, got {tensor.size()}"
        print(error_msg)
        raise ValueError(error_msg)