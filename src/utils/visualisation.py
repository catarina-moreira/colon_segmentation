import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib
import torch
from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image

def create_custom_colormap(name="tumor", start_color=(1, 0.7, 0.2), end_color=(1, 0, 0), N=256):
    """
    Create a custom colormap for tumor segmentation visualization.
    
    Args:
        name: Name of the colormap
        start_color: Starting color in RGB format
        end_color: Ending color in RGB format
        N: Number of color levels
        
    Returns:
        Matplotlib colormap
    """
    # Create a dictionary with color positions and corresponding RGB tuples
    colors = {0: start_color, 1: end_color}
    
    # Create a colormap from these colors
    cm = LinearSegmentedColormap.from_list(name, list(colors.values()), N=N)
    
    return cm

def visualize_slice(image, prediction=None, ground_truth=None, slice_idx=None, axis=0, figsize=(12, 4), alpha=0.5):
    """
    Visualize a slice of a 3D volume with optional prediction and ground truth overlays.
    
    Args:
        image: 3D image array (C, D, H, W) or (D, H, W)
        prediction: Optional 3D binary prediction array
        ground_truth: Optional 3D binary ground truth array
        slice_idx: Index of the slice to display (if None, will try to find a slice with segmentation)
        axis: Axis along which to take the slice (0=z, 1=y, 2=x)
        figsize: Figure size
        alpha: Transparency of segmentation overlays
        
    Returns:
        Matplotlib figure
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if prediction is not None and isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if ground_truth is not None and isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    
    # Handle channel dimension if present
    if image.ndim == 4:
        image = image[0]  # Take first channel
    if prediction is not None and prediction.ndim == 4:
        prediction = prediction[0]
    if ground_truth is not None and ground_truth.ndim == 4:
        ground_truth = ground_truth[0]
    
    # If slice_idx is not provided, try to find a slice with segmentation
    if slice_idx is None:
        if ground_truth is not None:
            # Find a slice with ground truth
            slice_indices = np.where(np.sum(ground_truth, axis=(1, 2)) > 0)[0]
            if len(slice_indices) > 0:
                slice_idx = slice_indices[len(slice_indices) // 2]  # Middle slice with segmentation
            else:
                slice_idx = image.shape[0] // 2  # Middle slice
        elif prediction is not None:
            # Find a slice with prediction
            slice_indices = np.where(np.sum(prediction, axis=(1, 2)) > 0)[0]
            if len(slice_indices) > 0:
                slice_idx = slice_indices[len(slice_indices) // 2]  # Middle slice with segmentation
            else:
                slice_idx = image.shape[0] // 2  # Middle slice
        else:
            # No segmentation available, use middle slice
            slice_idx = image.shape[0] // 2
    
    # Take slices based on the specified axis
    if axis == 0:  # z-axis (axial)
        img_slice = image[slice_idx]
        pred_slice = prediction[slice_idx] if prediction is not None else None
        gt_slice = ground_truth[slice_idx] if ground_truth is not None else None
        orientation = "Axial"
    elif axis == 1:  # y-axis (coronal)
        img_slice = image[:, slice_idx, :]
        pred_slice = prediction[:, slice_idx, :] if prediction is not None else None
        gt_slice = ground_truth[:, slice_idx, :] if ground_truth is not None else None
        orientation = "Coronal"
    elif axis == 2:  # x-axis (sagittal)
        img_slice = image[:, :, slice_idx]
        pred_slice = prediction[:, :, slice_idx] if prediction is not None else None
        gt_slice = ground_truth[:, :, slice_idx] if ground_truth is not None else None
        orientation = "Sagittal"
    
    # Create figure based on what segmentations are available
    if ground_truth is not None and prediction is not None:
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        titles = ["Image", "Prediction", "Ground Truth"]
    elif ground_truth is not None or prediction is not None:
        fig, axs = plt.subplots(1, 2, figsize=(figsize[0] * 2/3, figsize[1]))
        titles = ["Image", "Ground Truth" if ground_truth is not None else "Prediction"]
    else:
        fig, axs = plt.subplots(1, 1, figsize=(figsize[0] * 1/3, figsize[1]))
        axs = [axs]
        titles = ["Image"]
    
    # Plot image
    axs[0].imshow(img_slice, cmap='gray')
    axs[0].set_title(f"{orientation} View - {titles[0]}")
    axs[0].axis('off')
    
    # Custom colormaps
    tumor_cmap = create_custom_colormap("tumor", (1, 0.7, 0.2), (1, 0, 0))
    gt_cmap = create_custom_colormap("ground_truth", (0.2, 0.7, 1), (0, 0, 1))
    
    # Plot prediction overlay if available
    if prediction is not None and ground_truth is not None:
        axs[1].imshow(img_slice, cmap='gray')
        pred_mask = np.ma.masked_where(pred_slice == 0, pred_slice)
        axs[1].imshow(pred_mask, cmap=tumor_cmap, alpha=alpha)
        axs[1].set_title(f"{orientation} View - {titles[1]}")
        axs[1].axis('off')
        
        axs[2].imshow(img_slice, cmap='gray')
        gt_mask = np.ma.masked_where(gt_slice == 0, gt_slice)
        axs[2].imshow(gt_mask, cmap=gt_cmap, alpha=alpha)
        axs[2].set_title(f"{orientation} View - {titles[2]}")
        axs[2].axis('off')
    elif prediction is not None:
        axs[1].imshow(img_slice, cmap='gray')
        pred_mask = np.ma.masked_where(pred_slice == 0, pred_slice)
        axs[1].imshow(pred_mask, cmap=tumor_cmap, alpha=alpha)
        axs[1].set_title(f"{orientation} View - {titles[1]}")
        axs[1].axis('off')
    elif ground_truth is not None:
        axs[1].imshow(img_slice, cmap='gray')
        gt_mask = np.ma.masked_where(gt_slice == 0, gt_slice)
        axs[1].imshow(gt_mask, cmap=gt_cmap, alpha=alpha)
        axs[1].set_title(f"{orientation} View - {titles[1]}")
        axs[1].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_volume_montage(image, prediction=None, ground_truth=None, n_slices=5, axis=0, figsize=(15, 10)):
    """
    Visualize multiple slices from a volume as a montage.
    
    Args:
        image: 3D image array (C, D, H, W) or (D, H, W)
        prediction: Optional 3D binary prediction array
        ground_truth: Optional 3D binary ground truth array
        n_slices: Number of slices to display
        axis: Axis along which to take slices (0=z, 1=y, 2=x)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if prediction is not None and isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if ground_truth is not None and isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    
    # Handle channel dimension if present
    if image.ndim == 4:
        image = image[0]  # Take first channel
    if prediction is not None and prediction.ndim == 4:
        prediction = prediction[0]
    if ground_truth is not None and ground_truth.ndim == 4:
        ground_truth = ground_truth[0]
    
    # Find slices with segmentation if available
    slice_indices = None
    
    if ground_truth is not None:
        if axis == 0:
            sums = np.sum(ground_truth, axis=(1, 2))
        elif axis == 1:
            sums = np.sum(ground_truth, axis=(0, 2))
        else:
            sums = np.sum(ground_truth, axis=(0, 1))
        
        non_zero_indices = np.where(sums > 0)[0]
        
        if len(non_zero_indices) > 0:
            # Evenly sample from non-zero slices
            if len(non_zero_indices) >= n_slices:
                step = len(non_zero_indices) // n_slices
                slice_indices = non_zero_indices[::step][:n_slices]
            else:
                # If fewer non-zero slices than requested, use all and pad with adjacent slices
                slice_indices = np.array(sorted(list(set(non_zero_indices))))
                
                while len(slice_indices) < n_slices:
                    # Add slices before and after
                    new_indices = []
                    for idx in slice_indices:
                        if idx - 1 >= 0 and idx - 1 not in slice_indices:
                            new_indices.append(idx - 1)
                        if idx + 1 < image.shape[axis] and idx + 1 not in slice_indices:
                            new_indices.append(idx + 1)
                        if len(set(list(slice_indices) + new_indices)) >= n_slices:
                            break
                    
                    if not new_indices:
                        break
                    
                    slice_indices = np.array(sorted(list(set(list(slice_indices) + new_indices))))[:n_slices]
    
    if slice_indices is None or len(slice_indices) < n_slices:
        # If no segmentation or not enough slices with segmentation, evenly sample the volume
        total_slices = image.shape[axis]
        step = total_slices // n_slices
        slice_indices = np.arange(0, total_slices, step)[:n_slices]
    
    # Create the montage figure
    n_cols = min(5, n_slices)
    n_rows = (n_slices + n_cols - 1) // n_cols
    
    # Create figure based on what segmentations are available
    if ground_truth is not None and prediction is not None:
        fig, axs = plt.subplots(n_rows * 3, n_cols, figsize=figsize)
        # Reshape axs for easier indexing if it's a 1D array
        if n_rows * 3 == 1 or n_cols == 1:
            axs = axs.reshape(n_rows * 3, n_cols)
    else:
        fig, axs = plt.subplots(n_rows * 2, n_cols, figsize=figsize)
        # Reshape axs for easier indexing if it's a 1D array
        if n_rows * 2 == 1 or n_cols == 1:
            axs = axs.reshape(n_rows * 2, n_cols)
    
    # Custom colormaps
    tumor_cmap = create_custom_colormap("tumor", (1, 0.7, 0.2), (1, 0, 0))
    gt_cmap = create_custom_colormap("ground_truth", (0.2, 0.7, 1), (0, 0, 1))
    
    # Determine orientation label
    orientation = "Axial" if axis == 0 else "Coronal" if axis == 1 else "Sagittal"
    
    # Plot slices
    for i, slice_idx in enumerate(slice_indices):
        row = i // n_cols
        col = i % n_cols
        
        # Extract slices based on the specified axis
        if axis == 0:  # z-axis (axial)
            img_slice = image[slice_idx]
            pred_slice = prediction[slice_idx] if prediction is not None else None
            gt_slice = ground_truth[slice_idx] if ground_truth is not None else None
        elif axis == 1:  # y-axis (coronal)
            img_slice = image[:, slice_idx, :]
            pred_slice = prediction[:, slice_idx, :] if prediction is not None else None
            gt_slice = ground_truth[:, slice_idx, :] if ground_truth is not None else None
        elif axis == 2:  # x-axis (sagittal)
            img_slice = image[:, :, slice_idx]
            pred_slice = prediction[:, :, slice_idx] if prediction is not None else None
            gt_slice = ground_truth[:, :, slice_idx] if ground_truth is not None else None
        
        # Plot image
        if ground_truth is not None and prediction is not None:
            # Image
            axs[row * 3, col].imshow(img_slice, cmap='gray')
            axs[row * 3, col].set_title(f"{orientation} {slice_idx}")
            axs[row * 3, col].axis('off')
            
            # Prediction
            axs[row * 3 + 1, col].imshow(img_slice, cmap='gray')
            if pred_slice is not None:
                pred_mask = np.ma.masked_where(pred_slice == 0, pred_slice)
                axs[row * 3 + 1, col].imshow(pred_mask, cmap=tumor_cmap, alpha=0.5)
            axs[row * 3 + 1, col].set_title("Prediction")
            axs[row * 3 + 1, col].axis('off')
            
            # Ground Truth
            axs[row * 3 + 2, col].imshow(img_slice, cmap='gray')
            if gt_slice is not None:
                gt_mask = np.ma.masked_where(gt_slice == 0, gt_slice)
                axs[row * 3 + 2, col].imshow(gt_mask, cmap=gt_cmap, alpha=0.5)
            axs[row * 3 + 2, col].set_title("Ground Truth")
            axs[row * 3 + 2, col].axis('off')
        else:
            # Image
            axs[row * 2, col].imshow(img_slice, cmap='gray')
            axs[row * 2, col].set_title(f"{orientation} {slice_idx}")
            axs[row * 2, col].axis('off')
            
            # Prediction or Ground Truth
            axs[row * 2 + 1, col].imshow(img_slice, cmap='gray')
            if prediction is not None and pred_slice is not None:
                pred_mask = np.ma.masked_where(pred_slice == 0, pred_slice)
                axs[row * 2 + 1, col].imshow(pred_mask, cmap=tumor_cmap, alpha=0.5)
                axs[row * 2 + 1, col].set_title("Prediction")
            elif ground_truth is not None and gt_slice is not None:
                gt_mask = np.ma.masked_where(gt_slice == 0, gt_slice)
                axs[row * 2 + 1, col].imshow(gt_mask, cmap=gt_cmap, alpha=0.5)
                axs[row * 2 + 1, col].set_title("Ground Truth")
            axs[row * 2 + 1, col].axis('off')
    
    # Hide any unused subplots
    for i in range(len(slice_indices), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        
        if ground_truth is not None and prediction is not None:
            axs[row * 3, col].axis('off')
            axs[row * 3 + 1, col].axis('off')
            axs[row * 3 + 2, col].axis('off')
        else:
            axs[row * 2, col].axis('off')
            axs[row * 2 + 1, col].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_3d_volume(image, prediction=None, ground_truth=None, threshold=0.5):
    """
    Visualize a 3D volume with optional segmentation overlays.
    Uses MONAI's matshow3d for 3D rendering.
    
    Args:
        image: 3D image array (C, D, H, W) or (D, H, W)
        prediction: Optional 3D prediction array (probability or binary)
        ground_truth: Optional 3D binary ground truth array
        threshold: Threshold for prediction if it's probability map
        
    Returns:
        MONAI 3D visualization
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if prediction is not None and isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if ground_truth is not None and isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    
    # Handle channel dimension if present
    if image.ndim == 4:
        image = image[0]  # Take first channel
    if prediction is not None and prediction.ndim == 4:
        prediction = prediction[0]
    if ground_truth is not None and ground_truth.ndim == 4:
        ground_truth = ground_truth[0]
    
    # Apply threshold to prediction if needed
    if prediction is not None and prediction.max() > 1:
        prediction = (prediction > threshold).astype(np.float32)
    
    # Prepare visualization
    if ground_truth is not None and prediction is not None:
        # Blend image, prediction and ground truth
        # prediction in red, ground truth in blue
        pred_colored = np.zeros((*prediction.shape, 3), dtype=np.float32)
        pred_colored[..., 0] = prediction  # Red channel
        
        gt_colored = np.zeros((*ground_truth.shape, 3), dtype=np.float32)
        gt_colored[..., 2] = ground_truth  # Blue channel
        
        # Normalize image to [0, 1] for blending
        image_norm = (image - image.min()) / (image.max() - image.min())
        
        # Create RGB image
        image_rgb = np.stack([image_norm, image_norm, image_norm], axis=-1)
        
        # Blend
        blended = blend_images(image_rgb, pred_colored, alpha=0.5)
        blended = blend_images(blended, gt_colored, alpha=0.5)
        
        return matshow3d(blended, figsize=(10, 10), every_n=5)
    
    elif prediction is not None:
        # Blend image and prediction
        # Prediction in red
        pred_colored = np.zeros((*prediction.shape, 3), dtype=np.float32)
        pred_colored[..., 0] = prediction  # Red channel
        
        # Normalize image to [0, 1] for blending
        image_norm = (image - image.min()) / (image.max() - image.min())
        
        # Create RGB image
        image_rgb = np.stack([image_norm, image_norm, image_norm], axis=-1)
        
        # Blend
        blended = blend_images(image_rgb, pred_colored, alpha=0.5)
        
        return matshow3d(blended, figsize=(10, 10), every_n=5)
    
    elif ground_truth is not None:
        # Blend image and ground truth
        # Ground truth in blue
        gt_colored = np.zeros((*ground_truth.shape, 3), dtype=np.float32)
        gt_colored[..., 2] = ground_truth  # Blue channel
        
        # Normalize image to [0, 1] for blending
        image_norm = (image - image.min()) / (image.max() - image.min())
        
        # Create RGB image
        image_rgb = np.stack([image_norm, image_norm, image_norm], axis=-1)
        
        # Blend
        blended = blend_images(image_rgb, gt_colored, alpha=0.5)
        
        return matshow3d(blended, figsize=(10, 10), every_n=5)
    
    else:
        # Just show the image
        return matshow3d(image, figsize=(10, 10), every_n=5)

def plot_training_curves(metrics_dict, figsize=(12, 8), save_path=None):
    """
    Plot training and validation metrics curves.
    
    Args:
        metrics_dict: Dictionary with metrics (e.g., {'train_loss': [...], 'val_dice': [...]})
        figsize: Figure size
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib figure
    """
    # Determine number of metrics to plot
    n_metrics = len(metrics_dict)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, (metric_name, metric_values) in enumerate(metrics_dict.items()):
        axes[i].plot(metric_values, marker='.', linestyle='-')
        axes[i].set_title(metric_name)
        axes[i].grid(True, alpha=0.3)
        
        # Find best value and mark it
        if 'loss' in metric_name.lower():
            best_epoch = np.argmin(metric_values)
            best_value = metric_values[best_epoch]
            label = 'Min'
        else:
            best_epoch = np.argmax(metric_values)
            best_value = metric_values[best_epoch]
            label = 'Max'
        
        axes[i].plot(best_epoch, best_value, 'ro')
        axes[i].annotate(f'{label}: {best_value:.4f}',
                         xy=(best_epoch, best_value),
                         xytext=(10, 0),
                         textcoords='offset points')
    
    # Set common x-axis label
    plt.xlabel('Epochs')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig

def visualize_prediction_error(image, prediction, ground_truth, slice_idx=None, axis=0, figsize=(15, 5)):
    """
    Visualize prediction errors (false positives and false negatives).
    
    Args:
        image: 3D image array (C, D, H, W) or (D, H, W)
        prediction: 3D binary prediction array
        ground_truth: 3D binary ground truth array
        slice_idx: Index of the slice to display (if None, will find a slice with segmentation)
        axis: Axis along which to take the slice (0=z, 1=y, 2=x)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    
    # Handle channel dimension if present
    if image.ndim == 4:
        image = image[0]  # Take first channel
    if prediction.ndim == 4:
        prediction = prediction[0]
    if ground_truth.ndim == 4:
        ground_truth = ground_truth[0]
    
    # Ensure binary masks
    prediction = (prediction > 0).astype(np.uint8)
    ground_truth = (ground_truth > 0).astype(np.uint8)
    
    # Calculate error masks
    false_positive = np.logical_and(prediction == 1, ground_truth == 0)
    false_negative = np.logical_and(prediction == 0, ground_truth == 1)
    true_positive = np.logical_and(prediction == 1, ground_truth == 1)
    
    # If slice_idx is not provided, try to find a slice with segmentation
    if slice_idx is None:
        # Find a slice with either false positives or false negatives
        error_sum = false_positive + false_negative
        
        if axis == 0:
            error_slices = np.sum(error_sum, axis=(1, 2))
        elif axis == 1:
            error_slices = np.sum(error_sum, axis=(0, 2))
        else:
            error_slices = np.sum(error_sum, axis=(0, 1))
        
        error_indices = np.where(error_slices > 0)[0]
        
        if len(error_indices) > 0:
            slice_idx = error_indices[len(error_indices) // 2]  # Middle slice with errors
        else:
            # No errors found, try to find a slice with ground truth
            if axis == 0:
                gt_slices = np.sum(ground_truth, axis=(1, 2))
            elif axis == 1:
                gt_slices = np.sum(ground_truth, axis=(0, 2))
            else:
                gt_slices = np.sum(ground_truth, axis=(0, 1))
            
            gt_indices = np.where(gt_slices > 0)[0]
            
            if len(gt_indices) > 0:
                slice_idx = gt_indices[len(gt_indices) // 2]  # Middle slice with ground truth
            else:
                slice_idx = image.shape[axis] // 2  # Middle slice
    
    # Take slices based on the specified axis
    if axis == 0:  # z-axis (axial)
        img_slice = image[slice_idx]
        tp_slice = true_positive[slice_idx]
        fp_slice = false_positive[slice_idx]
        fn_slice = false_negative[slice_idx]
        orientation = "Axial"
    elif axis == 1:  # y-axis (coronal)
        img_slice = image[:, slice_idx, :]
        tp_slice = true_positive[:, slice_idx, :]
        fp_slice = false_positive[:, slice_idx, :]
        fn_slice = false_negative[:, slice_idx, :]
        orientation = "Coronal"
    elif axis == 2:  # x-axis (sagittal)
        img_slice = image[:, :, slice_idx]
        tp_slice = true_positive[:, :, slice_idx]
        fp_slice = false_positive[:, :, slice_idx]
        fn_slice = false_negative[:, :, slice_idx]
        orientation = "Sagittal"
    
    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    # Plot original image
    axs[0].imshow(img_slice, cmap='gray')
    axs[0].set_title(f"{orientation} View - Original Image")
    axs[0].axis('off')
    
    # Plot prediction with true positives (green), false positives (red)
    axs[1].imshow(img_slice, cmap='gray')
    
    # Create mask for true positives
    tp_mask = np.ma.masked_where(tp_slice == 0, tp_slice)
    axs[1].imshow(tp_mask, cmap='Greens', alpha=0.5)
    
    # Create mask for false positives
    fp_mask = np.ma.masked_where(fp_slice == 0, fp_slice)
    axs[1].imshow(fp_mask, cmap='Reds', alpha=0.5)
    
    axs[1].set_title(f"Prediction\nGreen: True Positive, Red: False Positive")
    axs[1].axis('off')
    
    # Plot ground truth with true positives (green), false negatives (blue)
    axs[2].imshow(img_slice, cmap='gray')
    
    # Create mask for true positives
    axs[2].imshow(tp_mask, cmap='Greens', alpha=0.5)
    
    # Create mask for false negatives
    fn_mask = np.ma.masked_where(fn_slice == 0, fn_slice)
    axs[2].imshow(fn_mask, cmap='Blues', alpha=0.5)
    
    axs[2].set_title(f"Ground Truth\nGreen: True Positive, Blue: False Negative")
    axs[2].axis('off')
    
    plt.tight_layout()
    return fig

def save_visualization(fig, save_path, dpi=300):
    """
    Save a matplotlib figure to a file.
    
    Args:
        fig: Matplotlib figure
        save_path: Path to save the figure
        dpi: Resolution in dots per inch
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def create_visualization_grid(image_list, prediction_list=None, ground_truth_list=None, slice_indices=None, 
                              row_titles=None, col_titles=None, figsize=(15, 10)):
    """
    Create a grid of visualizations for multiple samples.
    
    Args:
        image_list: List of 3D image arrays
        prediction_list: Optional list of prediction arrays
        ground_truth_list: Optional list of ground truth arrays
        slice_indices: Optional list of slice indices (if None, will find slices with segmentation)
        row_titles: Optional list of row titles
        col_titles: Optional list of column titles
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_rows = len(image_list)
    n_cols = 3 if prediction_list is not None and ground_truth_list is not None else 2
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Ensure axs is a 2D array
    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    if n_cols == 1:
        axs = np.expand_dims(axs, axis=1)
    
    # Custom colormaps
    tumor_cmap = create_custom_colormap("tumor", (1, 0.7, 0.2), (1, 0, 0))
    gt_cmap = create_custom_colormap("ground_truth", (0.2, 0.7, 1), (0, 0, 1))
    
    # Set column titles if provided
    if col_titles is not None:
        for j, title in enumerate(col_titles[:n_cols]):
            axs[0, j].set_title(title, fontsize=12, fontweight='bold')
    
    # Process each row
    for i, image in enumerate(image_list):
        # Convert tensors to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        # Handle channel dimension if present
        if image.ndim == 4:
            image = image[0]
        
        # Get prediction and ground truth if available
        pred = None
        gt = None
        
        if prediction_list is not None and i < len(prediction_list):
            pred = prediction_list[i]
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
            if pred.ndim == 4:
                pred = pred[0]
        
        if ground_truth_list is not None and i < len(ground_truth_list):
            gt = ground_truth_list[i]
            if isinstance(gt, torch.Tensor):
                gt = gt.detach().cpu().numpy()
            if gt.ndim == 4:
                gt = gt[0]
        
        # Determine slice index
        slice_idx = None
        if slice_indices is not None and i < len(slice_indices):
            slice_idx = slice_indices[i]
        else:
            # Find a slice with segmentation
            if gt is not None:
                gt_sum = np.sum(gt, axis=(1, 2))
                gt_indices = np.where(gt_sum > 0)[0]
                if len(gt_indices) > 0:
                    slice_idx = gt_indices[len(gt_indices) // 2]
            
            if slice_idx is None and pred is not None:
                pred_sum = np.sum(pred, axis=(1, 2))
                pred_indices = np.where(pred_sum > 0)[0]
                if len(pred_indices) > 0:
                    slice_idx = pred_indices[len(pred_indices) // 2]
            
            if slice_idx is None:
                slice_idx = image.shape[0] // 2
        
        # Get slices
        img_slice = image[slice_idx]
        pred_slice = pred[slice_idx] if pred is not None else None
        gt_slice = gt[slice_idx] if gt is not None else None
        
        # Set row title if provided
        if row_titles is not None and i < len(row_titles):
            axs[i, 0].set_ylabel(row_titles[i], fontsize=12, fontweight='bold', rotation=0, ha='right', va='center')
        
        # Plot image
        axs[i, 0].imshow(img_slice, cmap='gray')
        axs[i, 0].set_title(f"Slice {slice_idx}" if i == 0 and col_titles is None else "")
        axs[i, 0].axis('off')
        
        # Plot prediction if available
        if pred is not None:
            axs[i, 1].imshow(img_slice, cmap='gray')
            if pred_slice is not None:
                pred_mask = np.ma.masked_where(pred_slice == 0, pred_slice)
                axs[i, 1].imshow(pred_mask, cmap=tumor_cmap, alpha=0.5)
            axs[i, 1].set_title("Prediction" if i == 0 and col_titles is None else "")
            axs[i, 1].axis('off')
        
        # Plot ground truth if available
        if gt is not None and n_cols > 2:
            axs[i, 2].imshow(img_slice, cmap='gray')
            if gt_slice is not None:
                gt_mask = np.ma.masked_where(gt_slice == 0, gt_slice)
                axs[i, 2].imshow(gt_mask, cmap=gt_cmap, alpha=0.5)
            axs[i, 2].set_title("Ground Truth" if i == 0 and col_titles is None else "")
            axs[i, 2].axis('off')
    
    plt.tight_layout()
    return fig
    
 