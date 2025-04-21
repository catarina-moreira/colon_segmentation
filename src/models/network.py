import torch
import torch.nn as nn
import monai
from monai.networks.nets import UNet, BasicUNet, DynUNet
from monai.networks.layers import Norm

class ModelFactory:
    """Factory class for creating segmentation models."""
    
    @staticmethod
    def create_model(model_type, in_channels=1, out_channels=2, dimensions=3):
        """
        Create a segmentation model based on the specified type.
        
        Args:
            model_type: Type of model ('unet', 'basicunet', 'dynunet')
            in_channels: Number of input channels
            out_channels: Number of output channels
            dimensions: Number of spatial dimensions (2D or 3D)
            
        Returns:
            A PyTorch model
        """
        if model_type == 'unet':
            return UNetModel(in_channels, out_channels, dimensions)
        elif model_type == 'basicunet':
            return BasicUNetModel(in_channels, out_channels, dimensions)
        elif model_type == 'dynunet':
            return DynUNetModel(in_channels, out_channels, dimensions)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class UNetModel(nn.Module):
    """Standard 3D UNet model."""
    
    def __init__(self, in_channels=1, out_channels=2, dimensions=3):
        super().__init__()
        
        self.net = UNet(
            dimensions=dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
    
    def forward(self, x):
        return self.net(x)


class BasicUNetModel(nn.Module):
    """MONAI's lightweight UNet model."""
    
    def __init__(self, in_channels=1, out_channels=2, dimensions=3):
        super().__init__()
        
        self.net = BasicUNet(
            dimensions=dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            features=(32, 64, 128, 256, 512),
        )
    
    def forward(self, x):
        return self.net(x)


class DynUNetModel(nn.Module):
    """DynUNet model (similar to nnUNet architecture)."""
    
    def __init__(self, in_channels=1, out_channels=2, dimensions=3):
        super().__init__()
        
        # Define kernel sizes, strides, and upsample kernel sizes
        # These can also be dynamically computed based on input image size
        # as done in the original nnUNet implementation
        kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        upsample_kernel_size = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        
        self.net = DynUNet(
            spatial_dims=dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            norm_name="instance",
            deep_supervision=True,
            res_block=True,
        )
    
    def forward(self, x):
        return self.net(x)


class EnsembleModel(nn.Module):
    """
    Ensemble model that combines predictions from multiple models.
    
    The ensemble can use simple averaging or weighted averaging
    based on the provided weights for each model.
    """
    
    def __init__(self, models, weights=None):
        """
        Initialize ensemble model.
        
        Args:
            models: Dictionary mapping model_type -> model
            weights: Optional dictionary of model_type -> weight
                     If None, all models are weighted equally.
        """
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.weights = weights if weights is not None else {k: 1.0 for k in models.keys()}
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.weights.values())
        self.weights = {k: v / weight_sum for k, v in self.weights.items()}
    
    def forward(self, x):
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted average of model outputs
        """
        outputs = []
        
        for name, model in self.models.items():
            # Run forward pass through the model
            output = model(x)
            
            # Handle deep supervision outputs for DynUNet
            if isinstance(output, list):
                output = output[0]  # Take final output
            
            # Apply softmax to get probabilities
            output = torch.softmax(output, dim=1)
            
            # Add to outputs list with appropriate weight
            outputs.append((output, self.weights[name]))
        
        # Compute weighted sum
        weighted_sum = sum(output * weight for output, weight in outputs)
        
        return weighted_sum


# Additional utility functions for model initialization and loading

def get_device():
    """Get the available computation device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_trained_model(model_type, checkpoint_path, in_channels=1, out_channels=2, dimensions=3, device=None):
    if device is None:
        device = get_device()
    
    model = ModelFactory.create_model(model_type, in_channels, out_channels, dimensions)
    
    # Load checkpoint and handle both direct state_dict and nested dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Handle checkpoint saved with save_model function
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Handle direct state_dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

def create_ensemble(model_types, checkpoint_paths, weights=None, in_channels=1, out_channels=2, dimensions=3, device=None):
    """
    Create an ensemble of models from checkpoints.
    
    Args:
        model_types: List of model types to include in the ensemble
        checkpoint_paths: Dictionary mapping model_type -> checkpoint_path
        weights: Optional dictionary of model_type -> weight
        in_channels: Number of input channels
        out_channels: Number of output channels
        dimensions: Number of spatial dimensions
        device: Computation device (if None, will use get_device())
    
    Returns:
        Ensemble model on the specified device
    """
    if device is None:
        device = get_device()
    
    models = {}
    for model_type in model_types:
        if model_type in checkpoint_paths:
            model = load_trained_model(
                model_type, 
                checkpoint_paths[model_type],
                in_channels,
                out_channels,
                dimensions,
                device
            )
            models[model_type] = model
    
    if not models:
        raise ValueError("No valid models found for ensemble")
    
    return EnsembleModel(models, weights).to(device)

def initialize_model(model_type, in_channels=1, out_channels=2, dimensions=3, device=None):
    """
    Initialize a model and move it to the specified device.
    
    Args:
        model_type: Type of model to initialize
        in_channels: Number of input channels
        out_channels: Number of output channels
        dimensions: Number of spatial dimensions
        device: Computation device (if None, will use get_device())
    
    Returns:
        Initialized model on the specified device
    """
    if device is None:
        device = get_device()
    
    model = ModelFactory.create_model(model_type, in_channels, out_channels, dimensions)
    return model.to(device)
 