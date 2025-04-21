import os
import argparse
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm
import nibabel as nib
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import utility modules
from utils.data_utils import prepare_datalist, get_transforms, remove_small_components, save_prediction
from utils.model_utils import predict_with_sliding_window, predict_with_tta
from utils.visualization import visualize_slice, save_visualization
from models.network import create_ensemble, load_trained_model

from monai.metrics import SurfaceDistanceMetric

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ensemble inference for colon cancer segmentation")
    parser.add_argument("--config", type=str, default="configs/ensemble_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument("--model_dir", type=str, help="Path to model directory with checkpoints")
    parser.add_argument("--device", type=str, help="Device to use (cuda, cuda:0, cpu)")
    parser.add_argument("--use_tta", action="store_true", help="Use test-time augmentation")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def update_config(config, args):
    """Update configuration with command line arguments."""
    for arg, value in vars(args).items():
        if value is not None and arg != "config":
            if arg in config:
                config[arg] = value
            elif "." in arg:
                # Handle nested config parameters
                parts = arg.split(".")
                curr = config
                for part in parts[:-1]:
                    if part not in curr:
                        curr[part] = {}
                    curr = curr[part]
                curr[parts[-1]] = value
    return config

def setup_output_dirs(output_dir):
    """Set up output directories."""
    os.makedirs(output_dir, exist_ok=True)
    predictions_dir = os.path.join(output_dir, "predictions")
    visualizations_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    return predictions_dir, visualizations_dir

def load_models(config):
    """Load trained models for ensemble."""
    models = {}
    device = torch.device(config["device"])
    
    for model_type in config["models_to_ensemble"]:
        model_path = os.path.join(config["model_dir"], f"best_model_{model_type}.pth")
        
        if os.path.exists(model_path):
            print(f"Loading {model_type} model from {model_path}")
            try:
                model = load_trained_model(
                    model_type=model_type,
                    checkpoint_path=model_path,
                    in_channels=config["in_channels"],
                    out_channels=config["out_channels"],
                    dimensions=config["dimensions"],
                    device=device
                )
                models[model_type] = model
            except Exception as e:
                print(f"Error loading {model_type} model: {e}")
        else:
            print(f"Warning: Model checkpoint for {model_type} not found at {model_path}")
    
    if not models:
        raise ValueError("No models could be loaded for ensemble. Please check paths and model types.")

    # Print ensemble weights (normalization will happen in EnsembleModel)
    print("Using ensemble weights:")
    for model_type in models.keys():
        weight = config["model_weights"].get(model_type, 1.0)
        print(f"  {model_type}: {weight:.3f}")
    
    return models

def create_ensemble_model(models, weights, device):
    """Create an ensemble model."""
    import torch.nn as nn
    
    class EnsembleModel(nn.Module):
        def __init__(self, models, weights):
            super().__init__()
            self.models = nn.ModuleDict(models)
            self.weights = weights
            
            # Normalize weights to sum to 1
            weight_sum = sum(self.weights.values())
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}
        
        def forward(self, x):
            outputs = []
            
            for name, model in self.models.items():
                # Run forward pass through the model
                output = model(x)
                
                # Handle deep supervision outputs for DynUNet
                if isinstance(output, list):
                    output = output[0]  # Take final output
                
                # Add to outputs list with appropriate weight
                outputs.append((output, self.weights[name]))
            
            # Compute weighted sum
            weighted_sum = sum(output * weight for output, weight in outputs)
            
            return weighted_sum
    
    # Create and return the ensemble model
    return EnsembleModel(models, weights).to(device)

def run_ensemble_inference(ensemble_model, test_loader, config, predictions_dir, visualizations_dir):
    """Run inference using ensemble model."""
    device = torch.device(config["device"])
    ensemble_model.eval()
    
    # Track metrics if ground truth is available
    has_labels = "label" in next(iter(test_loader))
    
    if has_labels:
        from monai.metrics import DiceMetric, compute_hausdorff_distance, compute_average_surface_distance
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        all_metrics = {
            "dice": [],
            "hausdorff": [],
            "surface_distance": []
        }
        # Create metrics log file
        with open(os.path.join(config["output_dir"], "metrics_per_case.csv"), "w") as f:
            f.write("filename,dice_score,hausdorff95,avg_surface_distance\n")
    
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(test_loader, desc="Running inference")):
            # Get inputs
            test_inputs = batch_data["image"].to(device)
            
            # Get original metadata
            test_meta = batch_data["image_meta_dict"]
            orig_img_path = test_meta["filename_or_obj"]
            filename = os.path.basename(orig_img_path)
            
            # Run inference
            if config["use_tta"]:
                # Use test-time augmentation
                outputs = predict_with_tta(
                    model=model,
                    image=inputs,
                    roi_size=config["roi_size"],
                    sw_batch_size=config["sw_batch_size"],
                    overlap=config["overlap"],
                    memory_efficient=config.get("memory_efficient_tta", False)
                )
            else:
                # Standard sliding window inference
                outputs = predict_with_sliding_window(
                    model=ensemble_model,
                    image=test_inputs,
                    roi_size=config["roi_size"],
                    sw_batch_size=config["sw_batch_size"],
                    overlap=config["overlap"]
                )
            
            # Apply softmax to get probabilities
            outputs_softmax = torch.softmax(outputs, dim=1)
            
            # Get binary prediction (argmax)
            prediction = torch.argmax(outputs_softmax, dim=1).cpu().numpy().astype(np.uint8)
            
            # Apply post-processing
            if config["apply_post_processing"]:
                # Get voxel spacing from the image
                if "pixdim" in test_meta:
                    spacing = test_meta["pixdim"][1:4]  # Get voxel spacing
                else:
                    spacing = (1.5, 1.5, 2.0)  # Default spacing
                
                # Remove small isolated components
                for b in range(prediction.shape[0]):
                    prediction[b] = remove_small_components(
                        prediction[b], 
                        spacing, 
                        config["min_tumor_size_mm3"]
                    )
            
            # Save prediction
            save_prediction(
                prediction=prediction[0],
                original_image_path=orig_img_path,
                output_path=os.path.join(predictions_dir, filename)
            )
            
            # Calculate metrics if ground truth is available
            if has_labels:
                label = batch_data["label"].to(device)
                
                # Convert prediction to tensor
                pred_tensor = torch.from_numpy(prediction).unsqueeze(1).to(device)
                
                # Compute Dice score
                dice_metric(y_pred=pred_tensor, y=label)
                dice_val = dice_metric.aggregate().item()
                dice_metric.reset()
                all_metrics["dice"].append(dice_val)
                
                # Compute Hausdorff distance and surface distance
                hausdorff_val = compute_hausdorff_distance(
                    y_pred=pred_tensor, y=label, 
                    include_background=False, percentile=95
                ).cpu().numpy()[0][0]
                
                surface_metric = SurfaceDistanceMetric(include_background=False, reduction="mean")
                surface_metric(y_pred=pred_tensor, y=label)
                surface_dist_val = surface_metric.aggregate().item()
                surface_metric.reset()
                
                all_metrics["hausdorff"].append(hausdorff_val)
                all_metrics["surface_distance"].append(surface_dist_val)
                
                # Write metrics to file
                with open(os.path.join(config["output_dir"], "metrics_per_case.csv"), "a") as f:
                    f.write(f"{filename},{dice_val:.4f},{hausdorff_val:.4f},{surface_dist_val:.4f}\n")
            
            # Visualize first few cases
            if idx < config["num_visualizations"]:
                # Create visualization
                fig = visualize_slice(
                    image=test_inputs[0].cpu().numpy(),
                    prediction=prediction[0],
                    ground_truth=batch_data["label"][0].cpu().numpy() if has_labels else None,
                    slice_idx=None,
                    axis=0,
                    figsize=(15, 5) if has_labels else (10, 5)
                )
                
                # Save visualization
                save_visualization(
                    fig=fig,
                    save_path=os.path.join(visualizations_dir, f"vis_{filename[:-7]}.png"),
                    dpi=150
                )
    
    # Compute and return mean metrics
    if has_labels:
        mean_metrics = {
            "dice": np.mean(all_metrics["dice"]),
            "hausdorff": np.mean(all_metrics["hausdorff"]),
            "surface_distance": np.mean(all_metrics["surface_distance"])
        }
        
        # Print and save overall metrics
        metrics_summary = (
            f"Overall Metrics:\n"
            f"Mean Dice Score: {mean_metrics['dice']:.4f}\n"
            f"Mean Hausdorff Distance (95%): {mean_metrics['hausdorff']:.4f}\n"
            f"Mean Average Surface Distance: {mean_metrics['surface_distance']:.4f}\n"
        )
        
        print(metrics_summary)
        
        with open(os.path.join(config["output_dir"], "metrics_summary.txt"), "w") as f:
            f.write(metrics_summary)
        
        # Plot histogram of Dice scores
        plt.figure(figsize=(10, 6))
        plt.hist(all_metrics["dice"], bins=20, alpha=0.7, color='skyblue')
        plt.axvline(mean_metrics["dice"], color='red', linestyle='--', label=f'Mean: {mean_metrics["dice"]:.4f}')
        plt.title('Distribution of Dice Scores Across Test Set')
        plt.xlabel('Dice Score')
        plt.ylabel('Number of Cases')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(config["output_dir"], "dice_histogram.png"))
        plt.close()
        
        return mean_metrics
    
    return None

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config = update_config(config, args)
    
    # Set up output directories
    predictions_dir, visualizations_dir = setup_output_dirs(config["output_dir"])
    
    # Save configuration
    with open(os.path.join(config["output_dir"], "ensemble_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Device setup
    device = torch.device(config["device"])
    print(f"Using device: {device}")
    
    # Prepare data
    print("Preparing test data...")
    test_dicts = prepare_datalist(data_dir=config["data_dir"], test=True)
    
    # Get transforms
    test_transforms = get_transforms(
        phase="test", 
        intensity_bounds=config["intensity_bounds"]
    )
    
    # Create test dataset and dataloader
    from monai.data import Dataset
    test_ds = Dataset(data=test_dicts, transform=test_transforms)
    
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available()
    )
    
    # Load trained models
    models = load_models(config)
    
    # Create ensemble model
    ensemble_model = create_ensemble_model(
        models=models,
        weights=config["model_weights"],
        device=device
    )
    
    # Run inference
    print("Running ensemble inference...")
    metrics = run_ensemble_inference(
        ensemble_model=ensemble_model,
        test_loader=test_loader,
        config=config,
        predictions_dir=predictions_dir,
        visualizations_dir=visualizations_dir
    )
    
    print(f"Ensemble inference complete! Results saved to {config['output_dir']}")
    
    # Save metrics if available
    if metrics:
        with open(os.path.join(config["output_dir"], "metrics.json"), "w") as f:
            json.dump(metrics, f)
    
    return 0

if __name__ == "__main__":
    main()
 