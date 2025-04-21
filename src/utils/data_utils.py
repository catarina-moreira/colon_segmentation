import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandRotate90d,
    RandShiftIntensityd,
    ToTensord,
    RandFlipd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandAdjustContrastd,
    Orientationd,
    Spacingd,
    SpatialPadd,
    RandAffined,
)

def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    monai.utils.set_determinism(seed=seed)

def prepare_datalist(data_dir, validation_split=0.2, test=False, seed=42):
    """
    Prepare dictionaries for training, validation and test datasets.
    
    Args:
        data_dir: Path to the dataset directory
        validation_split: Fraction of training data to use for validation
        test: If True, prepare test dataset instead of training/validation
        seed: Random seed for reproducibility
        
    Returns:
        If test=False: training and validation data dictionaries
        If test=True: test data dictionaries
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    if not test:
        # Get all training images and labels
        train_images = sorted([
            os.path.join(data_dir, "imagesTr", f) 
            for f in os.listdir(os.path.join(data_dir, "imagesTr"))
            if f.endswith('.nii.gz')
        ])
        
        train_labels = sorted([
            os.path.join(data_dir, "labelsTr", f) 
            for f in os.listdir(os.path.join(data_dir, "labelsTr"))
            if f.endswith('.nii.gz')
        ])
        
        # Create data dictionaries
        data_dicts = [
            {"image": img, "label": lbl}
            for img, lbl in zip(train_images, train_labels)
        ]
        
        # Shuffle data
        random.shuffle(data_dicts)
        
        # Split into training and validation
        val_size = int(validation_split * len(data_dicts))
        train_dicts = data_dicts[val_size:]
        val_dicts = data_dicts[:val_size]
        
        print(f"Training samples: {len(train_dicts)}, Validation samples: {len(val_dicts)}")
        
        return train_dicts, val_dicts
    else:
        # Get all test images
        test_images = sorted([
            os.path.join(data_dir, "imagesTs", f) 
            for f in os.listdir(os.path.join(data_dir, "imagesTs"))
            if f.endswith('.nii.gz')
        ])
        
        # Check if test labels are available
        test_label_dir = os.path.join(data_dir, "labelsTs")
        has_test_labels = os.path.exists(test_label_dir)
        
        if has_test_labels:
            test_labels = sorted([
                os.path.join(test_label_dir, f) 
                for f in os.listdir(test_label_dir)
                if f.endswith('.nii.gz')
            ])
            
            # Create data dictionaries with labels
            test_dicts = [
                {"image": img, "label": lbl}
                for img, lbl in zip(test_images, test_labels)
            ]
        else:
            # Create data dictionaries without labels
            test_dicts = [{"image": img} for img in test_images]
        
        print(f"Test samples: {len(test_dicts)}")
        
        return test_dicts

def get_transforms(
    phase="train", 
    intensity_bounds=(-175, 250), 
    spatial_size=(96, 96, 96),
    pos_sample_ratio=1.0
):
    """
    Create transform pipelines for different phases.
    
    Args:
        phase: "train", "val", or "test"
        intensity_bounds: Range of intensity values to scale (in HU for CT)
        spatial_size: Size of patches for training
        pos_sample_ratio: Ratio of positive to negative samples for patch sampling
        
    Returns:
        MONAI transforms for the specified phase
    """
    # Common transforms for all phases
    common_transforms = [
        LoadImaged(keys=["image", "label"] if phase != "test" else ["image"]),
        AddChanneld(keys=["image", "label"] if phase != "test" else ["image"]),
        Orientationd(
            keys=["image", "label"] if phase != "test" else ["image"], 
            axcodes="RAS"
        ),  # Ensure consistent orientation
        Spacingd(
            keys=["image", "label"] if phase != "test" else ["image"], 
            pixdim=(1.5, 1.5, 2.0), 
            mode=("bilinear", "nearest") if phase != "test" else "bilinear"
        ),  # Resample to common spacing
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_bounds[0],
            a_max=intensity_bounds[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
    
    if phase == "train":
        # Training transforms with augmentation
        return Compose(
            common_transforms + [
                # Crop background to focus on relevant regions
                CropForegroundd(
                    keys=["image", "label"], 
                    source_key="image", 
                    margin=10
                ),
                # Random patch sampling with balanced foreground/background
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=spatial_size,
                    pos=pos_sample_ratio,
                    neg=1.0 - pos_sample_ratio,
                    num_samples=4,
                ),
                # Data augmentation transforms
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                RandAffined(
                    keys=["image", "label"],
                    prob=0.5,
                    rotate_range=(np.pi/36, np.pi/36, np.pi/36),  # 5 degrees
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest"),
                    padding_mode="zeros",
                ),
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                RandGaussianNoised(keys=["image"], std=0.01, prob=0.2),
                RandAdjustContrastd(keys=["image"], prob=0.3),
                ToTensord(keys=["image", "label"]),
            ]
        )
    elif phase == "val":
        # Validation transforms (no augmentation)
        return Compose(
            common_transforms + [
                CropForegroundd(
                    keys=["image", "label"], 
                    source_key="image", 
                    margin=10
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )
    else:  # Test transforms
        if "label" in common_transforms[0].keys:
            # If test labels are available
            return Compose(
                common_transforms + [
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            # If test labels are not available
            return Compose(
                common_transforms + [
                    ToTensord(keys=["image"]),
                ]
            )

def create_data_loaders(
    data_dir, 
    batch_size=2, 
    val_batch_size=1, 
    num_workers=4, 
    cache_rate=1.0,
    validation_split=0.2,
    pin_memory=True,
    **transform_kwargs
):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for training
        val_batch_size: Batch size for validation
        num_workers: Number of worker processes for data loading
        cache_rate: Cache rate for CacheDataset
        validation_split: Fraction of training data to use for validation
        pin_memory: Pin memory for faster GPU transfer
        transform_kwargs: Additional arguments for transforms
        
    Returns:
        Train loader, validation loader
    """
    # Prepare data lists
    train_dicts, val_dicts = prepare_datalist(
        data_dir=data_dir, 
        validation_split=validation_split
    )
    
    # Get transforms
    train_transforms = get_transforms(phase="train", **transform_kwargs)
    val_transforms = get_transforms(phase="val", **transform_kwargs)
    
    # Create datasets
    train_ds = monai.data.CacheDataset(
        data=train_dicts, 
        transform=train_transforms,
        cache_rate=cache_rate, 
        num_workers=num_workers
    )
    
    val_ds = monai.data.CacheDataset(
        data=val_dicts, 
        transform=val_transforms, 
        cache_rate=cache_rate, 
        num_workers=num_workers
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=monai.data.list_data_collate, 
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=val_batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader

def create_test_loader(
    data_dir, 
    batch_size=1, 
    num_workers=4, 
    pin_memory=True,
    **transform_kwargs
):
    """
    Create a data loader for testing.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for testing
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster GPU transfer
        transform_kwargs: Additional arguments for transforms
        
    Returns:
        Test loader
    """
    # Prepare test data
    test_dicts = prepare_datalist(data_dir=data_dir, test=True)
    
    # Get test transforms
    test_transforms = get_transforms(phase="test", **transform_kwargs)
    
    # Create test dataset
    test_ds = monai.data.Dataset(
        data=test_dicts, 
        transform=test_transforms
    )
    
    # Create test loader
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return test_loader

def get_dataset_info(data_dir):
    """
    Get information about the dataset.
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        Dictionary with dataset information
    """
    info = {}
    
    # Count number of training samples
    train_dir = os.path.join(data_dir, "imagesTr")
    if os.path.exists(train_dir):
        info["num_training"] = len([f for f in os.listdir(train_dir) if f.endswith('.nii.gz')])
    else:
        info["num_training"] = 0
    
    # Count number of test samples
    test_dir = os.path.join(data_dir, "imagesTs")
    if os.path.exists(test_dir):
        info["num_test"] = len([f for f in os.listdir(test_dir) if f.endswith('.nii.gz')])
    else:
        info["num_test"] = 0
    
    # Check if test labels are available
    info["has_test_labels"] = os.path.exists(os.path.join(data_dir, "labelsTs"))
    
    # Get example image info
    if info["num_training"] > 0:
        example_path = os.path.join(train_dir, os.listdir(train_dir)[0])
        example_img = nib.load(example_path)
        info["image_shape"] = example_img.shape
        info["image_spacing"] = example_img.header.get_zooms()
        info["image_dtype"] = example_img.get_data_dtype()
    
    return info

def remove_small_components(binary_mask, spacing, min_size_mm3):
    """
    Remove small isolated components from binary segmentation mask.
    
    Args:
        binary_mask: 3D binary array
        spacing: Tuple of voxel spacing in mm
        min_size_mm3: Minimum component size to keep in mm³
    
    Returns:
        Processed binary mask with small components removed
    """
    from scipy import ndimage
    
    # Calculate voxel volume in mm³
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    
    # Compute minimum component size in voxels
    min_size_voxels = int(min_size_mm3 / voxel_volume)
    
    # Find connected components
    labeled_mask, num_components = ndimage.label(binary_mask)
    
    # Count voxels for each component
    component_sizes = np.bincount(labeled_mask.reshape(-1))
    
    # Create mask for components to keep
    keep = component_sizes >= min_size_voxels
    keep[0] = False  # Background should remain 0
    
    # Apply the mask to remove small components
    processed_mask = np.isin(labeled_mask, np.where(keep)[0]).astype(binary_mask.dtype)
    
    return processed_mask

def save_prediction(prediction, original_image_path, output_path):
    """
    Save a prediction as a NIfTI file, using metadata from the original image.
    
    Args:
        prediction: Binary prediction array (can be numpy array or torch tensor)
        original_image_path: Path to the original image
        output_path: Path to save the prediction
    """
    # Convert to numpy if tensor
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    
    # Ensure prediction is binary and correct data type
    prediction = prediction.astype(np.uint8)
    
    # Load original image to get affine and header
    orig_img = nib.load(original_image_path)
    
    # Create NIfTI image with same affine and header
    pred_nii = nib.Nifti1Image(prediction, orig_img.affine, orig_img.header)
    
    # Save prediction
    nib.save(pred_nii, output_path)
 