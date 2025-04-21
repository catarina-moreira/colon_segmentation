import os
import argparse
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib
from torch.utils.data import DataLoader

# Import utility modules
from utils.data_utils import prepare_datalist, get_transforms, remove_small_components, save_prediction
from utils.model_utils import predict_with_sliding_window, predict_with_tta
from utils.visualization import (
    visualize_slice, 
    visualize_volume_montage, 
    visualize_prediction_error, 
    save_visualization
)
from models.network import load_trained_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inference for colon cancer segmentation")
    parser.add_argument("--config", type=str, default="configs/inference_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, help="Model type (unet, basicunet, dynunet)")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
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
    
    subdirs = [
        "predictions",         # For segmentation masks
        "visualizations",      # For 2D visualizations
        "error_analysis",      # For error analysis visualizations
        "montages"             # For slice montages
    ]
    
    paths = {}
    for subdir in subdirs:
        path = os.path.join(output_dir, subdir)
        os.makedirs(path, exist_ok=True)
        paths[subdir] = path
    
    return paths

def run_inference(model, test_loader, config, output_dirs):
    """Run inference on test data."""
    device = torch.device(config["device"])
    model.eval()
    
    # Check if ground truth is available
    has_labels = "label" in next(iter(test_loader))
    
    # Metrics calculation if ground truth is available
    if has_labels:
        from monai.metrics import (
            DiceMetric, 
            compute_hausdorff_distance, 
            compute_average_surface_distance, 
            SurfaceDistanceMetric
        )
        
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        surface_metric = SurfaceDistanceMetric(include_background=False, reduction="mean")
        
        metrics_per_case = {
            "filename": [],
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
            inputs = batch_data["image"].to(device)
            
            # Get metadata
            meta_dict = batch_data["image_meta_dict"]
            filename = os.path.basename(meta_dict["filename_or_obj"])
            
            # Run inference
            if config["use_tta"]:
                outputs = predict_with_tta(
                    model=model,
                    image=inputs,
                    roi_size=config["roi_size"],
                    sw_batch_size=config["sw_batch_size"],
                    overlap=config["overlap"],
                    memory_efficient=config.get("memory_efficient_tta", False)
                )
            else:
                outputs = predict_with_sliding_window(
                    model=model,
                    image=inputs,
                    roi_size=config["roi_size"],
                    sw_batch_size=config["sw_batch_size"],
                    overlap=config["overlap"]
                )
            
            # Handle deep supervision outputs
            if isinstance(outputs, list):
                outputs = outputs[0]
            
            # Apply softmax to get probabilities
            outputs_softmax = torch.softmax(outputs, dim=1)
            
            # Get binary prediction (argmax)
            prediction = torch.argmax(outputs_softmax, dim=1).cpu().numpy().astype(np.uint8)
            
            # Apply post-processing if enabled
            if config["apply_post_processing"]:
                # Get voxel spacing
                if "pixdim" in meta_dict:
                    spacing = meta_dict["pixdim"][1:4]
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
                original_image_path=meta_dict["filename_or_obj"],
                output_path=os.path.join(output_dirs["predictions"], filename)
            )
            
            # Calculate metrics if ground truth is available
            if has_labels:
                label = batch_data["label"].to(device)
                
                # Convert prediction to tensor for metric calculation
                pred_tensor = torch.from_numpy(prediction).unsqueeze(1).to(device)
                
                # Compute Dice score
                dice_metric(y_pred=pred_tensor, y=label)
                dice_val = dice_metric.aggregate().item()
                dice_metric.reset()
                
                # Compute Hausdorff distance
                hausdorff_val = compute_hausdorff_distance(
                    y_pred=pred_tensor, y=label, 
                    include_background=False, percentile=95
                ).cpu().numpy()[0][0]
                
                # Compute average surface distance
                surface_metric(y_pred=pred_tensor, y=label)
                surface_dist_val = surface_metric.aggregate().item()
                surface_metric.reset()
                
                # Store metrics
                metrics_per_case["filename"].append(filename)
                metrics_per_case["dice"].append(dice_val)
                metrics_per_case["hausdorff"].append(hausdorff_val)
                metrics_per_case["surface_distance"].append(surface_dist_val)
                
                # Write to CSV
                with open(os.path.join(config["output_dir"], "metrics_per_case.csv"), "a") as f:
                    f.write(f"{filename},{dice_val:.4f},{hausdorff_val:.4f},{surface_dist_val:.4f}\n")
                
                # Create error analysis visualization
                if idx < config["num_error_visualizations"]:
                    fig = visualize_prediction_error(
                        image=inputs[0].cpu().numpy(),
                        prediction=prediction[0],
                        ground_truth=label[0].cpu().numpy(),
                        slice_idx=None,
                        axis=0,
                        figsize=(15, 5)
                    )
                    
                    save_visualization(
                        fig=fig,
                        save_path=os.path.join(output_dirs["error_analysis"], f"error_{filename[:-7]}.png"),
                        dpi=150
                    )
            
            # Create standard visualization
            if idx < config["num_visualizations"]:
                fig = visualize_slice(
                    image=inputs[0].cpu().numpy(),
                    prediction=prediction[0],
                    ground_truth=label[0].cpu().numpy() if has_labels else None,  # Add conditional check
                    slice_idx=None,
                    axis=0,
                    figsize=(15, 5) if has_labels else (10, 5)
                )
                
                save_visualization(
                    fig=fig,
                    save_path=os.path.join(output_dirs["visualizations"], f"vis_{filename[:-7]}.png"),
                    dpi=150
                )
                
                # Create multi-slice montage
                fig = visualize_volume_montage(
                    image=inputs[0].cpu().numpy(),
                    prediction=prediction[0],
                    ground_truth=label[0].cpu().numpy() if has_labels else None,
                    n_slices=config["montage_slices"],
                    axis=0,
                    figsize=(15, 10)
                )
                
                save_visualization(
                    fig=fig,
                    save_path=os.path.join(output_dirs["montages"], f"montage_{filename[:-7]}.png"),
                    dpi=150
                )
    
    # If ground truth is available, compute summary metrics and generate plots
    if has_labels:
        # Compute mean metrics
        mean_metrics = {
            "dice": float(np.mean(metrics_per_case["dice"])),
            "hausdorff": float(np.mean(metrics_per_case["hausdorff"])),
            "surface_distance": float(np.mean(metrics_per_case["surface_distance"]))
        }
        
        # Generate summary report
        summary = (
            f"Model: {config['model_type']}\n"
            f"Test cases: {len(metrics_per_case['dice'])}\n"
            f"Mean Dice score: {mean_metrics['dice']:.4f}\n"
            f"Mean Hausdorff distance (95%): {mean_metrics['hausdorff']:.4f}\n"
            f"Mean surface distance: {mean_metrics['surface_distance']:.4f}\n"
        )
        
        print(summary)
        
        with open(os.path.join(config["output_dir"], "metrics_summary.txt"), "w") as f:
            f.write(summary)
        
        # Save metrics to JSON
        with open(os.path.join(config["output_dir"], "metrics.json"), "w") as f:
            json.dump(mean_metrics, f, indent=4)
        
        # Generate histogram of Dice scores
        plt.figure(figsize=(10, 6))
        plt.hist(metrics_per_case["dice"], bins=20, alpha=0.7, color='skyblue')
        plt.axvline(mean_metrics["dice"], color='red', linestyle='--', 
                    label=f"Mean: {mean_metrics['dice']:.4f}")
        plt.title("Distribution of Dice Scores")
        plt.xlabel("Dice Score")
        plt.ylabel("Number of Cases")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config["output_dir"], "dice_histogram.png"), dpi=150)
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
    output_dirs = setup_output_dirs(config["output_dir"])
    
    # Save configuration
    with open(os.path.join(config["output_dir"], "inference_config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Set device
    device = torch.device(config["device"])
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading {config['model_type']} model from {config['model_path']}")
    model = load_trained_model(
        model_type=config["model_type"],
        checkpoint_path=config["model_path"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        dimensions=config["dimensions"],
        device=device
    )
    
    # Prepare data
    print("Preparing test data...")
    test_dicts = prepare_datalist(data_dir=config["data_dir"], test=True)
    
    # Get test transforms
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
    
    print(f"Running inference on {len(test_ds)} test cases...")
    
    # Run inference
    metrics = run_inference(
        model=model,
        test_loader=test_loader,
        config=config,
        output_dirs=output_dirs
    )
    
    print(f"Inference complete! Results saved to {config['output_dir']}")
    
    return 0

if __name__ == "__main__":
    main()
    
