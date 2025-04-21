import os
import argparse
import yaml
import time
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Import utility modules
from utils.data_utils import set_random_seed, create_data_loaders
from utils.model_utils import (
    create_loss_function, 
    create_optimizer, 
    create_scheduler,
    save_model,
    load_model,
    predict_with_sliding_window, 
    create_evaluation_metric
)
from utils.visualization import (
    visualize_slice,
    plot_training_curves,
    save_visualization
)
from models.network import initialize_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a segmentation model for colon cancer")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument("--model_type", type=str, help="Model type (unet, basicunet, dynunet)")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--device", type=str, help="Device to use (cuda, cuda:0, cpu)")
    parser.add_argument("--seed", type=int, help="Random seed")
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
                # Handle nested config parameters (e.g., "optimizer.lr")
                parts = arg.split(".")
                curr = config
                for part in parts[:-1]:
                    if part not in curr:
                        curr[part] = {}
                    curr = curr[part]
                curr[parts[-1]] = value
    return config

def create_experiment_dir(output_dir, model_type):
    """Create a unique experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"{model_type}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "visualizations"), exist_ok=True)
    return exp_dir

def save_config(config, exp_dir):
    """Save configuration to file."""
    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, config):
    """Train model for one epoch."""
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['max_epochs']}")
    
    for batch_idx, batch_data in enumerate(progress_bar):
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if config['model_type'] == 'dynunet' and hasattr(model, 'do_ds') and model.do_ds:
            # Handle deep supervision outputs
            outputs = model(inputs)
            
            # Use all deep supervision outputs with different weights
            if isinstance(outputs, list):
                # Apply loss to each output with descending weights
                ds_weights = [0.5 ** i for i in range(len(outputs))]
                ds_weights = [w / sum(ds_weights) for w in ds_weights]  # Normalize weights
                
                # Calculate weighted loss for each deep supervision output
                losses = [loss_fn(out, labels) * weight for out, weight in zip(outputs, ds_weights)]
                loss = sum(losses)
            else:
                # Fallback if output is not a list
                loss = loss_fn(outputs, labels)
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return epoch_loss / len(train_loader)

def validate(model, val_loader, loss_fn, metric_fn, device, config):
    """Validate the model."""
    model.eval()
    val_loss = 0
    metric_fn.reset()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            # Sliding window inference for validation
            outputs = predict_with_sliding_window(
                model, inputs, 
                roi_size=config['patch_size'], 
                sw_batch_size=config['sw_batch_size'],
                overlap=config['overlap']
            )
            
            # Calculate validation loss
            val_loss_batch = loss_fn(outputs, labels)
            val_loss += val_loss_batch.item()
            
            # Compute metric
            if config['model_type'] == 'dynunet' and isinstance(outputs, list):
                outputs = outputs[0]
            
            metric_fn(y_pred=outputs, y=labels)
    
    # Aggregate metrics
    metric = metric_fn.aggregate().item()
    metric_fn.reset()
    
    return val_loss / len(val_loader), metric

def train(config):
    """Train the model according to the configuration."""
    # Set device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    set_random_seed(config['seed'])
    
    # Create experiment directory
    exp_dir = create_experiment_dir(config['output_dir'], config['model_type'])
    print(f"Experiment directory: {exp_dir}")
    
    # Save configuration
    save_config(config, exp_dir)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        val_batch_size=config['val_batch_size'],
        num_workers=config['num_workers'],
        validation_split=config['validation_split'],
        pin_memory=True,
        spatial_size=config['patch_size'],
        pos_sample_ratio=config['pos_sample_ratio'],
        intensity_bounds=config['intensity_bounds']
    )
    
    # Create model
    model = initialize_model(
        model_type=config['model_type'],
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        dimensions=config['dimensions'],
        device=device
    )
    print(f"Created {config['model_type']} model with "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
          f"trainable parameters")
    
    # Create loss function
    loss_fn = create_loss_function(
        include_background=config['loss']['include_background'],
        to_onehot_y=config['loss']['to_onehot_y'],
        softmax=config['loss']['softmax'],
        sigmoid=config['loss']['sigmoid'],
        use_ce=config['loss']['use_ce']
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        learning_rate=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay'],
        optimizer_type=config['optimizer']['type']
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer=optimizer,
        mode=config['scheduler']['mode'],
        factor=config['scheduler']['factor'],
        patience=config['scheduler']['patience'],
        verbose=config['scheduler']['verbose']
    )
    
    # Create metric for validation
    metric_fn = create_evaluation_metric(
        include_background=config['metric']['include_background'],
        reduction=config['metric']['reduction']
    )
    
    # Initialize training variables
    start_epoch = 0
    best_metric = -1
    best_metric_epoch = -1
    best_model_path = os.path.join(exp_dir, "checkpoints", f"best_model_{config['model_type']}.pth")
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_metrics = []
    
    # Resume training if requested
    if config['resume'] and os.path.exists(best_model_path):
        print(f"Resuming training from {best_model_path}")
        model, optimizer, scheduler, start_epoch, best_metric = load_model(
            model=model,
            path=best_model_path,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        best_metric_epoch = start_epoch
    
    # Training loop
    print(f"Starting training from epoch {start_epoch + 1}")
    start_time = time.time()
    
    for epoch in range(start_epoch, config['max_epochs']):
        # Train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch + 1, config)
        train_losses.append(train_loss)
        
        print(f"Epoch {epoch + 1}/{config['max_epochs']}, Average Loss: {train_loss:.4f}")
        
        # Validate model
        if (epoch + 1) % config['val_interval'] == 0:
            val_loss, metric = validate(model, val_loader, loss_fn, metric_fn, device, config)
            val_losses.append(val_loss)
            val_metrics.append(metric)
            
            scheduler.step(val_loss)
            
            print(f"Validation Loss: {val_loss:.4f}, Metric: {metric:.4f}")
            
            # Save visualization of a validation sample
            if (epoch + 1) % config['vis_interval'] == 0:
                visualize_validation_sample(model, val_loader, device, exp_dir, epoch + 1, config)
            
            # Save best model
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                
                save_model(
                    model=model,
                    path=best_model_path,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=best_metric
                )
                
                print(f"New best model saved with metric: {best_metric:.4f} at epoch {best_metric_epoch}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % config['checkpoint_interval'] == 0:
                checkpoint_path = os.path.join(exp_dir, "checkpoints", f"checkpoint_epoch_{epoch + 1}.pth")
                save_model(
                    model=model,
                    path=checkpoint_path,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_metric=best_metric
                )
            
            # Early stopping
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Plot and save training curves
        if (epoch + 1) % config['plot_interval'] == 0:
            plot_training_curves(
                metrics_dict={
                    "Training Loss": train_losses,
                    "Validation Loss": val_losses if val_losses else None,
                    "Validation Metric": val_metrics if val_metrics else None
                },
                save_path=os.path.join(exp_dir, "training_curves.png")
            )
    
    # Training finished
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")
    
    # Save final training curves
    plot_training_curves(
        metrics_dict={
            "Training Loss": train_losses,
            "Validation Loss": val_losses,
            "Validation Metric": val_metrics
        },
        save_path=os.path.join(exp_dir, "training_curves.png")
    )
    
    # Save training history
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_metric": val_metrics,
        "best_metric": float(best_metric),
        "best_metric_epoch": int(best_metric_epoch),
        "total_time": float(total_time)
    }
    
    with open(os.path.join(exp_dir, "training_history.json"), "w") as f:
        json.dump(history, f)
    
    return exp_dir, best_model_path, best_metric

def visualize_validation_sample(model, val_loader, device, exp_dir, epoch, config):
    """Visualize a validation sample with model prediction."""
    model.eval()
    
    # Get a validation sample
    val_data = next(iter(val_loader))
    image = val_data["image"].to(device)
    label = val_data["label"].to(device)
    
    with torch.no_grad():
        # Sliding window inference
        prediction = predict_with_sliding_window(
            model, image, 
            roi_size=config['patch_size'], 
            sw_batch_size=config['sw_batch_size'],
            overlap=config['overlap']
        )
        
        # Apply softmax and get binary prediction
        if config['model_type'] == 'dynunet' and isinstance(prediction, list):
            prediction = prediction[0]
        
        prediction_softmax = torch.softmax(prediction, dim=1)
        prediction_binary = torch.argmax(prediction_softmax, dim=1, keepdim=True)
    
    # Visualize prediction
    fig = visualize_slice(
        image=image[0].detach().cpu().numpy(),
        prediction=prediction_binary[0].detach().cpu().numpy(),
        ground_truth=label[0].detach().cpu().numpy(),
        slice_idx=None,
        axis=0,
        figsize=(15, 5),
        alpha=0.5
    )
    
    # Save visualization
    save_visualization(
        fig=fig,
        save_path=os.path.join(exp_dir, "visualizations", f"val_pred_epoch_{epoch}.png"),
        dpi=150
    )

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config = update_config(config, args)
    
    # Train model
    exp_dir, best_model_path, best_metric = train(config)
    
    print(f"Training completed. Results saved to {exp_dir}")
    print(f"Best model saved at {best_model_path} with metric {best_metric:.4f}")
    
    return 0

if __name__ == "__main__":
    main()
    
