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

# Set default values
DATA_DIR="./data/Task10_Colon"
OUTPUT_DIR="./outputs"
RESULTS_DIR="./results"

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
mkdir -p ${OUTPUT_DIR}/models
mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${RESULTS_DIR}/inference
mkdir -p ${RESULTS_DIR}/ensemble

# Define timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/logs/pipeline_${TIMESTAMP}.log"
MODELS_DIR="${OUTPUT_DIR}/models"

# Train the models
train_models() {
    echo -e "${BLUE}Training models...${NC}"
    
    # Train DynUNet model
    echo -e "${GREEN}Training DynUNet model...${NC}"
    python src/train.py --config configs/train_config.yaml \
        --model_type dynunet \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --device ${DEVICE} 2>&1 | tee -a ${LOG_FILE}
    
    # Train UNet model
    echo -e "${GREEN}Training UNet model...${NC}"
    python src/train.py --config configs/train_config.yaml \
        --model_type unet \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --device ${DEVICE} 2>&1 | tee -a ${LOG_FILE}
    
    # Train BasicUNet model
    echo -e "${GREEN}Training BasicUNet model...${NC}"
    python src/train.py --config configs/train_config.yaml \
        --model_type basicunet \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --device ${DEVICE} 2>&1 | tee -a ${LOG_FILE}
    
    echo -e "${GREEN}All models trained successfully!${NC}"
}

# Find the latest model directory for a specific model type
find_latest_model_dir() {
    local model_type=$1
    local latest_dir=$(find ${OUTPUT_DIR} -type d -name "${model_type}_*" | sort -r | head -1)
    echo ${latest_dir}
}

# Get the best model path for a specific model type
get_best_model_path() {
    local model_type=$1
    local model_dir=$(find_latest_model_dir ${model_type})
    
    if [ -z "${model_dir}" ]; then
        echo -e "${RED}Error: No model directory found for ${model_type}${NC}"
        return 1
    fi
    
    local best_model_path="${model_dir}/checkpoints/best_model_${model_type}.pth"
    
    if [ ! -f "${best_model_path}" ]; then
        echo -e "${RED}Error: Best model not found at ${best_model_path}${NC}"
        return 1
    fi
    
    echo ${best_model_path}
}

# Run inference with individual models
run_inference() {
    echo -e "${BLUE}Running inference with individual models...${NC}"
    
    # Get best model paths
    DYNUNET_PATH=$(get_best_model_path dynunet)
    UNET_PATH=$(get_best_model_path unet)
    BASICUNET_PATH=$(get_best_model_path basicunet)
    
    # Check if model paths were found
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error getting model paths. Skipping inference.${NC}"
        return 1
    fi
    
    echo -e "Using models:"
    echo -e "  DynUNet: ${DYNUNET_PATH}"
    echo -e "  UNet: ${UNET_PATH}"
    echo -e "  BasicUNet: ${BASICUNET_PATH}"
    
    # Create model output directories
    mkdir -p ${MODELS_DIR}
    cp "${DYNUNET_PATH}" "${MODELS_DIR}/best_model_dynunet.pth"
    cp "${UNET_PATH}" "${MODELS_DIR}/best_model_unet.pth"
    cp "${BASICUNET_PATH}" "${MODELS_DIR}/best_model_basicunet.pth"
    
    # Run inference with DynUNet model
    echo -e "${GREEN}Running inference with DynUNet model...${NC}"
    python src/inference.py --config configs/inference_config.yaml \
        --model_type dynunet \
        --model_path "${DYNUNET_PATH}" \
        --data_dir ${DATA_DIR} \
        --output_dir "${RESULTS_DIR}/inference/dynunet" \
        --device ${DEVICE} \
        --use_tta 2>&1 | tee -a ${LOG_FILE}
    
    # Run inference with UNet model
    echo -e "${GREEN}Running inference with UNet model...${NC}"
    python src/inference.py --config configs/inference_config.yaml \
        --model_type unet \
        --model_path "${UNET_PATH}" \
        --data_dir ${DATA_DIR} \
        --output_dir "${RESULTS_DIR}/inference/unet" \
        --device ${DEVICE} \
        --use_tta 2>&1 | tee -a ${LOG_FILE}
    
    # Run inference with BasicUNet model
    echo -e "${GREEN}Running inference with BasicUNet model...${NC}"
    python src/inference.py --config configs/inference_config.yaml \
        --model_type basicunet \
        --model_path "${BASICUNET_PATH}" \
        --data_dir ${DATA_DIR} \
        --output_dir "${RESULTS_DIR}/inference/basicunet" \
        --device ${DEVICE} \
        --use_tta 2>&1 | tee -a ${LOG_FILE}
    
    echo -e "${GREEN}All inference runs completed!${NC}"
}

# Run ensemble inference
run_ensemble() {
    echo -e "${BLUE}Running ensemble inference...${NC}"
    
    # Check if models exist
    if [ ! -f "${MODELS_DIR}/best_model_dynunet.pth" ] || \
       [ ! -f "${MODELS_DIR}/best_model_unet.pth" ] || \
       [ ! -f "${MODELS_DIR}/best_model_basicunet.pth" ]; then
        echo -e "${RED}Error: Model files not found in ${MODELS_DIR}${NC}"
        return 1
    fi
    
    # Run ensemble inference
    python src/ensemble.py --config configs/ensemble_config.yaml \
        --model_dir "${MODELS_DIR}" \
        --data_dir ${DATA_DIR} \
        --output_dir "${RESULTS_DIR}/ensemble" \
        --device ${DEVICE} \
        --use_tta 2>&1 | tee -a ${LOG_FILE}
    
    echo -e "${GREEN}Ensemble inference completed!${NC}"
}

# Compare results
compare_results() {
    echo -e "${BLUE}Comparing model results...${NC}"
    
    # Check if metrics files exist
    if [ ! -f "${RESULTS_DIR}/inference/dynunet/metrics_summary.txt" ] || \
       [ ! -f "${RESULTS_DIR}/inference/unet/metrics_summary.txt" ] || \
       [ ! -f "${RESULTS_DIR}/inference/basicunet/metrics_summary.txt" ] || \
       [ ! -f "${RESULTS_DIR}/ensemble/metrics_summary.txt" ]; then
        echo -e "${RED}Error: Metrics files not found. Cannot compare results.${NC}"
        return 1
    fi
    
    # Extract Dice scores from the results
    DYNUNET_DICE=$(grep "Mean Dice" ${RESULTS_DIR}/inference/dynunet/metrics_summary.txt | awk '{print $4}')
    UNET_DICE=$(grep "Mean Dice" ${RESULTS_DIR}/inference/unet/metrics_summary.txt | awk '{print $4}')
    BASICUNET_DICE=$(grep "Mean Dice" ${RESULTS_DIR}/inference/basicunet/metrics_summary.txt | awk '{print $4}')
    ENSEMBLE_DICE=$(grep "Mean Dice" ${RESULTS_DIR}/ensemble/metrics_summary.txt | awk '{print $4}')
    
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
    echo -e "\nResults Summary:" > ${RESULTS_DIR}/summary.txt
    echo -e "  DynUNet: Dice = ${DYNUNET_DICE}" >> ${RESULTS_DIR}/summary.txt
    echo -e "  UNet: Dice = ${UNET_DICE}" >> ${RESULTS_DIR}/summary.txt
    echo -e "  BasicUNet: Dice = ${BASICUNET_DICE}" >> ${RESULTS_DIR}/summary.txt
    echo -e "  Ensemble: Dice = ${ENSEMBLE_DICE}" >> ${RESULTS_DIR}/summary.txt
    echo -e "\nComparison with MSD Challenge:" >> ${RESULTS_DIR}/summary.txt
    echo -e "  MSD Challenge winner: Dice ≈ 0.56" >> ${RESULTS_DIR}/summary.txt
    echo -e "  Our best model: Dice = $(echo -e "$DYNUNET_DICE\n$UNET_DICE\n$BASICUNET_DICE\n$ENSEMBLE_DICE" | sort -nr | head -1)" >> ${RESULTS_DIR}/summary.txt
    
    echo -e "${GREEN}Results comparison completed and saved to ${RESULTS_DIR}/summary.txt${NC}"
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
