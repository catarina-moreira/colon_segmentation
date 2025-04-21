import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

def create_loss_function(include_background=False, to_onehot_y=True, softmax=True, sigmoid=False, use_ce=True):
    """
    Create a loss function for segmentation.
    
    Args:
        include_background: Whether to include background in loss calculation
        to_onehot_y: Whether to convert target to one-hot encoding
        softmax: Whether to apply softmax to predictions
        sigmoid: Whether to apply sigmoid to predictions
        use_ce: Whether to use cross-entropy loss with Dice loss
        
    Returns:
        Loss function
    """
    if use_ce:
        return DiceCELoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            sigmoid=sigmoid
        )
    else:
        return DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            sigmoid=sigmoid
        )

def create_optimizer(model, learning_rate=3e-4, weight_decay=1e-5, optimizer_type="adamw"):
    """
    Create an optimizer for training.
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        optimizer_type: Type of optimizer ('adam', 'adamw', or 'sgd')
        
    Returns:
        PyTorch optimizer
    """
    if optimizer_type.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def create_scheduler(optimizer, mode="min", factor=0.5, patience=20, verbose=True):
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        mode: 'min' for decreasing metrics (loss), 'max' for increasing metrics (accuracy)
        factor: Factor by which to reduce learning rate
        patience: Number of epochs with no improvement after which to reduce LR
        verbose: Whether to print message when LR is reduced
        
    Returns:
        PyTorch learning rate scheduler
    """
    return ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        verbose=verbose
    )

def save_model(model, path, epoch=None, optimizer=None, scheduler=None, best_metric=None):
    """
    Save a model checkpoint.
    
    Args:
        model: PyTorch model
        path: Path to save the checkpoint
        epoch: Current epoch (optional)
        optimizer: PyTorch optimizer (optional)
        scheduler: PyTorch scheduler (optional)
        best_metric: Best validation metric (optional)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    
    if epoch is not None:
        checkpoint["epoch"] = epoch
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if best_metric is not None:
        checkpoint["best_metric"] = best_metric
    
    # Save checkpoint
    torch.save(checkpoint, path)

def load_model(model, path, optimizer=None, scheduler=None, device=None):
    """
    Load a model checkpoint.
    
    Args:
        model: PyTorch model
        path: Path to the checkpoint
        optimizer: PyTorch optimizer (optional)
        scheduler: PyTorch scheduler (optional)
        device: Computation device (if None, will use model's device)
        
    Returns:
        Tuple of (model, optimizer, scheduler, epoch, best_metric)
    """
    # Determine device
    if device is None:
        device = next(model.parameters()).device
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Get epoch and best metric if available
    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", 0.0)
    
    return model, optimizer, scheduler, epoch, best_metric

def predict_with_sliding_window(model, image, roi_size, sw_batch_size=4, overlap=0.5, mode="gaussian"):
    """
    Run sliding window inference on an image.
    
    Args:
        model: PyTorch model
        image: Input tensor (B, C, H, W, D)
        roi_size: Size of sliding window (e.g., [96, 96, 96])
        sw_batch_size: Batch size for sliding window
        overlap: Amount of overlap between windows
        mode: Blending mode for overlapping windows
        
    Returns:
        Prediction tensor
    """
    model.eval()
    with torch.no_grad():
        prediction = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode=mode
        )
    
    return prediction

def predict_with_tta(model, image, roi_size, sw_batch_size=4, overlap=0.5, mode="gaussian", memory_efficient=False):
    """
    Run inference with test-time augmentation (TTA).
    
    Args:
        model: PyTorch model
        image: Input tensor (B, C, H, W, D)
        roi_size: Size of sliding window (e.g., [96, 96, 96])
        sw_batch_size: Batch size for sliding window
        overlap: Amount of overlap between windows
        mode: Blending mode for overlapping windows
        memory_efficient: If True, uses a memory-efficient approach by processing one augmentation at a time
        
    Returns:
        Prediction tensor after test-time augmentation
    """
    model.eval()
    
    with torch.no_grad():
        # Original prediction
        pred_orig = predict_with_sliding_window(model, image, roi_size, sw_batch_size, overlap, mode)
        
        if memory_efficient:
            # Memory-efficient version: accumulate running sum and count
            prediction_sum = pred_orig.clone()
            count = 1
            
            # Flip along depth (z-axis)
            image_z = torch.flip(image, dims=[2])
            pred_z = predict_with_sliding_window(model, image_z, roi_size, sw_batch_size, overlap, mode)
            pred_z = torch.flip(pred_z, dims=[2])
            prediction_sum += pred_z
            count += 1
            del pred_z, image_z  # Free memory
            
            # Flip along height (y-axis)
            image_y = torch.flip(image, dims=[3])
            pred_y = predict_with_sliding_window(model, image_y, roi_size, sw_batch_size, overlap, mode)
            pred_y = torch.flip(pred_y, dims=[3])
            prediction_sum += pred_y
            count += 1
            del pred_y, image_y  # Free memory
            
            # Flip along width (x-axis)
            image_x = torch.flip(image, dims=[4])
            pred_x = predict_with_sliding_window(model, image_x, roi_size, sw_batch_size, overlap, mode)
            pred_x = torch.flip(pred_x, dims=[4])
            prediction_sum += pred_x
            count += 1
            del pred_x, image_x  # Free memory
            
            # Average predictions
            prediction = prediction_sum / count
        else:
            # Standard version: stack all predictions
            # Flip along depth (z-axis)
            image_z = torch.flip(image, dims=[2])
            pred_z = predict_with_sliding_window(model, image_z, roi_size, sw_batch_size, overlap, mode)
            pred_z = torch.flip(pred_z, dims=[2])
            
            # Flip along height (y-axis)
            image_y = torch.flip(image, dims=[3])
            pred_y = predict_with_sliding_window(model, image_y, roi_size, sw_batch_size, overlap, mode)
            pred_y = torch.flip(pred_y, dims=[3])
            
            # Flip along width (x-axis)
            image_x = torch.flip(image, dims=[4])
            pred_x = predict_with_sliding_window(model, image_x, roi_size, sw_batch_size, overlap, mode)
            pred_x = torch.flip(pred_x, dims=[4])
            
            # Average predictions
            prediction = torch.stack([pred_orig, pred_z, pred_y, pred_x]).mean(dim=0)
        
    return prediction



def create_evaluation_metric(include_background=False, reduction="mean"):
    """
    Create a metric for evaluation.
    
    Args:
        include_background: Whether to include background in metric calculation
        reduction: Reduction method ('mean', 'sum', 'none')
        
    Returns:
        MONAI metric
    """
    return DiceMetric(include_background=include_background, reduction=reduction)

def post_process_predictions(predictions, threshold=0.5, argmax=True):
    """
    Post-process model predictions.
    
    Args:
        predictions: Model predictions (probabilities)
        threshold: Threshold for binary predictions (used if argmax=False)
        argmax: Whether to use argmax or thresholding
        
    Returns:
        Post-processed predictions
    """
    if argmax:
        # Apply argmax for multi-class segmentation
        post_process = AsDiscrete(argmax=True)
    else:
        # Apply thresholding for binary segmentation
        post_process = AsDiscrete(threshold=threshold)
    
    return post_process(predictions)

def calculate_flops(model, input_shape=(1, 1, 96, 96, 96)):
    """
    Calculate FLOPs (floating point operations) for a model.
    
    Args:
        model: PyTorch model
        input_shape: Input shape (B, C, H, W, D)
        
    Returns:
        Total FLOPs
    """
    try:
        from thop import profile
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)
        
        # Calculate FLOPs
        flops, params = profile(model, inputs=(dummy_input,))
        
        return {"flops": flops, "params": params}
    except ImportError:
        print("thop package not found. Please install it with pip install thop")
        return {"flops": 0, "params": sum(p.numel() for p in model.parameters() if p.requires_grad)}

def initialize_model_weights(model, initialization_method="kaiming_normal"):
    """
    Initialize model weights.
    
    Args:
        model: PyTorch model
        initialization_method: Method for weight initialization
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            if initialization_method == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif initialization_method == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif initialization_method == "xavier_normal":
                nn.init.xavier_normal_(m.weight)
            elif initialization_method == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm3d, nn.InstanceNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def get_learning_rate(optimizer):
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
