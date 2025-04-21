#!/bin/bash

# Define color codes for prettier output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print script header
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}     Colon Cancer Segmentation Pipeline     ${NC}"
echo -e "${BLUE}============================================${NC}"

# Check for CUDA availability
if python -c "import torch; print(torch.cuda.is_available());" | grep -q "True"; then
    DEVICE="cuda"
    echo -e "${GREEN}CUDA is available. Using GPU.${NC}"
else
    DEVICE="cpu"
    echo -e "${YELLOW}CUDA is not available. Using CPU. This will be slow!${NC}"
fi

# Create directories
echo -e "${BLUE}Creating directories...${NC}"
mkdir -p outputs/models
mkdir -p outputs/logs
mkdir -p results/inference
mkdir -p results/ensemble

# Define timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="outputs/logs/pipeline_${TIMESTAMP}.log"

# Train the models
train_models() {
    echo -e "${BLUE}Training models...${NC}"
    
    # Train DynUNet model
    echo -e "${GREEN}Training DynUNet model...${NC}"
    python src/train.py --config configs/train_config.yaml --model_type dynunet --device $DEVICE 2>&1 | tee -a $LOG_FILE
    
    # Train UNet model
    echo -e "${GREEN}Training UNet model...${NC}"
    python src/train.py --config configs/train_config.yaml --model_type unet --device $DEVICE 2>&1 | tee -a $LOG_FILE
    
    # Train BasicUNet model
    echo -e "${GREEN}Training BasicUNet model...${NC}"
    python src/train.py --config configs/train_config.yaml --model_type basicunet --device $DEVICE 2>&1 | tee -a $LOG_FILE
    
    echo -e "${GREEN}All models trained successfully!${NC}"
}

# Run inference with individual models
run_inference() {
    echo -e "${BLUE}Running inference with individual models...${NC}"
    
    # Find the most recent model folders
    DYNUNET_DIR=$(ls -td outputs/dynunet_* | head -1)
    UNET_DIR=$(ls -td outputs/unet_* | head -1)
    BASICUNET_DIR=$(ls -td outputs/basicunet_* | head -1)
    
    echo -e "Using models from:"
    echo -e "  DynUNet: ${DYNUNET_DIR}"
    echo -e "  UNet: ${UNET_DIR}"
    echo -e "  BasicUNet: ${BASICUNET_DIR}"
    
    # Run inference with DynUNet model
    echo -e "${GREEN}Running inference with DynUNet model...${NC}"
    python src/inference.py --config configs/inference_config.yaml \
        --model_type dynunet \
        --model_path "${DYNUNET_DIR}/checkpoints/best_model_dynunet.pth" \
        --output_dir "results/inference/dynunet" \
        --device $DEVICE \
        --use_tta 2>&1 | tee -a $LOG_FILE
    
    # Run inference with UNet model
    echo -e "${GREEN}Running inference with UNet model...${NC}"
    python src/inference.py --config configs/inference_config.yaml \
        --model_type unet \
        --model_path "${UNET_DIR}/checkpoints/best_model_unet.pth" \
        --output_dir "results/inference/unet" \
        --device $DEVICE \
        --use_tta 2>&1 | tee -a $LOG_FILE
    
    # Run inference with BasicUNet model
    echo -e "${GREEN}Running inference with BasicUNet model...${NC}"
    python src/inference.py --config configs/inference_config.yaml \
        --model_type basicunet \
        --model_path "${BASICUNET_DIR}/checkpoints/best_model_basicunet.pth" \
        --output_dir "results/inference/basicunet" \
        --device $DEVICE \
        --use_tta 2>&1 | tee -a $LOG_FILE
    
    echo -e "${GREEN}All inference runs completed!${NC}"
}

# Run ensemble inference
run_ensemble() {
    echo -e "${BLUE}Running ensemble inference...${NC}"
    
    # Prepare model directory with best models
    mkdir -p outputs/models
    
    # Find the most recent model folders
    DYNUNET_DIR=$(ls -td outputs/dynunet_* | head -1)
    UNET_DIR=$(ls -td outputs/unet_* | head -1)
    BASICUNET_DIR=$(ls -td outputs/basicunet_* | head -1)
    
    # Copy best models to models directory
    cp "${DYNUNET_DIR}/checkpoints/best_model_dynunet.pth" outputs/models/
    cp "${UNET_DIR}/checkpoints/best_model_unet.pth" outputs/models/
    cp "${BASICUNET_DIR}/checkpoints/best_model_basicunet.pth" outputs/models/
    
    # Run ensemble inference
    python src/ensemble.py --config configs/ensemble_config.yaml \
        --model_dir "outputs/models" \
        --output_dir "results/ensemble" \
        --device $DEVICE \
        --use_tta 2>&1 | tee -a $LOG_FILE
    
    echo -e "${GREEN}Ensemble inference completed!${NC}"
}

# Compare results
compare_results() {
    echo -e "${BLUE}Comparing model results...${NC}"
    
    # Extract Dice scores from the results
    DYNUNET_DICE=$(grep "Mean Dice" results/inference/dynunet/metrics_summary.txt | awk '{print $4}')
    UNET_DICE=$(grep "Mean Dice" results/inference/unet/metrics_summary.txt | awk '{print $4}')
    BASICUNET_DICE=$(grep "Mean Dice" results/inference/basicunet/metrics_summary.txt | awk '{print $4}')
    ENSEMBLE_DICE=$(grep "Mean Dice" results/ensemble/metrics_summary.txt | awk '{print $4}')
    
    echo -e "${YELLOW}Results Summary:${NC}"
    echo -e "  DynUNet: Dice = ${DYNUNET_DICE}"
    echo -e "  UNet: Dice = ${UNET_DICE}"
    echo -e "  BasicUNet: Dice = ${BASICUNET_DICE}"
    echo -e "  Ensemble: Dice = ${ENSEMBLE_DICE}"
    
    # Compare with best DSC from the MSD challenge paper
    echo -e "\n${YELLOW}Comparison with MSD Challenge:${NC}"
    echo -e "  MSD Challenge winner: Dice ≈ 0.56"
    echo -e "  Our best model: Dice = $(echo -e "$DYNUNET_DICE\n$UNET_DICE\n$BASICUNET_DICE\n$ENSEMBLE_DICE" | sort -nr | head -1)"
    
    # Save results summary
    echo -e "\nResults Summary:" > results/summary.txt
    echo -e "  DynUNet: Dice = ${DYNUNET_DICE}" >> results/summary.txt
    echo -e "  UNet: Dice = ${UNET_DICE}" >> results/summary.txt
    echo -e "  BasicUNet: Dice = ${BASICUNET_DICE}" >> results/summary.txt
    echo -e "  Ensemble: Dice = ${ENSEMBLE_DICE}" >> results/summary.txt
    echo -e "\nComparison with MSD Challenge:" >> results/summary.txt
    echo -e "  MSD Challenge winner: Dice ≈ 0.56" >> results/summary.txt
    echo -e "  Our best model: Dice = $(echo -e "$DYNUNET_DICE\n$UNET_DICE\n$BASICUNET_DICE\n$ENSEMBLE_DICE" | sort -nr | head -1)" >> results/summary.txt
    
    echo -e "${GREEN}Results comparison completed and saved to results/summary.txt${NC}"
}

# Main execution flow
main() {
    # Train models
    train_models
    
    # Run inference with individual models
    run_inference
    
    # Run ensemble inference
    run_ensemble
    
    # Compare results
    compare_results
    
    echo -e "${GREEN}Pipeline completed successfully!${NC}"
    echo -e "${BLUE}Log file: ${LOG_FILE}${NC}"
}

# Execute main function
main
