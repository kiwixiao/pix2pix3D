import logging
import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
from sklearn.metrics import f1_score, jaccard_score
from scipy.ndimage import zoom

def setup_logger():
    """Set up and return a logger"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def resample_image(image, new_spacing=[1.0, 1.0, 1.0]):
    """Resample the image to a new spacing
       This ensures all images are at the same scale before processing.
    """
    original_spacing = image.GetSpacing() # this image should be a return of sitk.ReadImage()

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

def extract_roi(image, mask=None, margin=20):
    """
    extract region of interest from the image and mask
    this focuses the model on the relevant part of the image, reducing unnecessary computation.
    """
    if mask is None:
        # if no mask is provided, use the entire image
        return image, None
    # find the non zero elements in the image, not the mask
    z, y, x = np.where(image > image.min())
    if len(z) == 0:
        # if no non-zero elements found in the image, return the entire image and mask
        return image, mask, (0,image.shape[0], 0, image.shape[1], 0, image.shape[2])

    z_min, z_max = max(z_min() - margin, 0), min(z.max() + margin, mask.shape[0])
    y_min, y_max = max(y_min() - margin, 0), min(y.max() + margin, mask.shape[1])
    x_min, x_max = max(x_min() - margin, 0), min(x.max() + margin, mask.shape[2])

    roi_image = image[z_min:z_max, y_min:y_max, x_min:x_max]
    roi_mask = mask[z_min:z_max, y_min:y_max, x_min:x_max] if mask is not None else None

    return roi_image, roi_mask, (z_min, z_max, y_min, y_max, x_min, x_max)

def resize_volume(img, target_shape):
    """
    resize a 3D volume to the target shape, this ensures all inputs to the model have the same dimensions
    """
    current_shape = img.shape
    factors = [float(t) / float(s) for t, s in zip(target_shape, current_shape)]
    return zoom(img, factors, order=1, mode='constant')

def preprocess_data(image_path, mask_path, target_shape, new_spacing=[1.0, 1.0, 1.0]):
    """
    prepricess the image and mask data.
    """
    # load image and mask
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path) if mask_path else None
    # resample to isotropic resolution
    image_resampled = resample_image(image, new_spacing)
    mask_resampled = resample_image(mask, new_spacing) if mask else None
    # convert to numpy arrays
    image_array = sitk.GetArrayFromImage(image_resampled)
    mask_array = sitk.GetArrayFromImage(mask_resampled) if mask else None
    # extract ROI
    #roi_image, roi_mask, roi_coords = extract_roi(image_array, mask_array)
    # normalized roi image to [-1, 1]
    #roi_image = 2 * (roi_image - roi_image.min()) / (roi_image.max() - roi_image.min()) - 1
    

    # resize ROI to rarget shape
    #roi_image_resized = resize_volume(roi_image, target_shape)
    #roi_mask_resized = resize_volume(roi_mask, target_shape) if roi_mask is not None else None
# Normalize image to [-1, 1]
    image_array = 2 * (image_array - image_array.min()) / (image_array.max() - image_array.min()) - 1
    
    # Resize to target shape
    image_resized = resize_volume(image_array, target_shape)
    mask_resized = resize_volume(mask_array, target_shape) if mask_array is not None else None
    
    return image_resized, mask_resized


def extract_patches(image, mask, patch_size, stride):
    # extract patches from a 3D image
    # image is numpy.ndarray
    # patch_size is tuple
    # stride is tuple: (depth, height, width)
    patches = []
    mask_patches = []
    for z in range(0, image.shape[0] - patch_size[0] + 1, stride[0]):
        for y in range(0, image.shape[1] - patch_size[1] + 1, stride[1]):
            for x in range(0, image.shape[2] - patch_size[2] + 1, stride[2]):
                patch = image[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                mask_patch = mask[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]]
                patches.append(patch)
                mask_patches.append(mask_patch)

    return patches, mask_patches

class PatchDataset(Dataset):
    def __init__(self, mri_images, mask_labels, patch_size=(64,64,64), stride=(32,32,32)):
        self.mri_images=mri_images
        self.mask_labels = mask_labels
        self.patch_size = patch_size
        self.stride = stride

        self.patches = []
        self.extract_all_patches = []

    def extract_all_patches(self):
        for mri, mask in zip(self.mri_images, self.mask_labels):
            mri_patches, mask_patches = extract_patches(mri, mask, self.patch_size, self.stride)
            self.patches.extend(list(zip(mri_patches, mask_patches)))

    def __len__(self):
        return len(self.mri_patches)

    def __getitem__(self, idx):
        img_patch, mask_patch = self.patches[idx]
        
        return torch.from_numpy(img_patch).float().unsqueeze(0), torch.from_numpy(mask_patch).float().unsqueeze(0)

def load_dataset(data_dir, target_shape, patch_size, stride, new_spacing=[1.0, 1.0, 1.0]):
    """
    load and preprocess all images in the data dir, this function prepares the entrire dataset for training.
    """
    logger = setup_logger()
    mri_images = []
    mask_labels = []
    # get all mri and maks files
    mri_files = sorted(glob.glob(os.path.join(data_dir, '*mri.nii*')))
    mask_files = sorted(glob.glob(os.path.join(data_dir, '*mask.nii*')))
    if len(mri_files) != len(mask_files):
        print(f"number of images {len(mri_files)}")
        print(f"number of masks {len(mask_files)}")
        
        raise ValueError("Number of MRI images and masks do not match.")
    # process each pair of MRI and mask
    for mri_file, mask_file in zip(mri_files,mask_files):
        logger.info(f"Processing {mri_file}")
        mri_image, mask_label = preprocess_data(mri_file, mask_file, target_shape, new_spacing)
        mri_images.append(mri_image)
        mask_labels.append(mask_label)
    dataset = PatchDataset(mri_images, mask_labels, patch_size, stride)

    return dataset

def calculate_metrics(true_mask, pred_mask):
    """
    calculate F1 score and IoU (Jaccard index) for the prediction
    These metrics help evaluate the performance of the segmentation model.
    """
    true_mask = true_mask.flatten()
    pred_mask = pred_mask.flatten()
    f1 = f1_score(true_mask, pred_mask > 0.5)
    iou = jaccard_score(true_mask, pred_mask > 0.5)
    return {'F1': f1, 'IoU': iou}



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
