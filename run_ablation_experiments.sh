#!/bin/bash
# Ablation Study Runner Script
# This script runs all ablation experiments sequentially

# Configuration
DATA_DIR="path/to/your/data"  # Update this path
BATCH_SIZE=8
NUM_WORKERS=8
LEARNING_RATE=0.001
EPOCHS=100
PATIENCE=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "HiFi-FD Ablation Study"
echo "=========================================="
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Data directory not found at $DATA_DIR${NC}"
    echo "Please update DATA_DIR in this script"
    exit 1
fi

# Create outputs directory
mkdir -p outputs

# Experiment 1: Full HiFi (Baseline)
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Experiment 1: Full HiFi${NC}"
echo -e "${YELLOW}========================================${NC}"
python src/train.py \
    --data_dir "$DATA_DIR" \
    --model_type full \
    --output_dir outputs/exp1_full \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --patience $PATIENCE

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Experiment 1 completed successfully${NC}"
else
    echo -e "${RED}Experiment 1 failed${NC}"
fi

# Experiment 2: RGB-Only Stream
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Experiment 2: RGB-Only Stream${NC}"
echo -e "${YELLOW}========================================${NC}"
python src/train.py \
    --data_dir "$DATA_DIR" \
    --model_type rgb_only \
    --output_dir outputs/exp2_rgb_only \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --patience $PATIENCE

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Experiment 2 completed successfully${NC}"
else
    echo -e "${RED}Experiment 2 failed${NC}"
fi

# Experiment 3: SRM-Only Stream
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Experiment 3: SRM-Only Stream${NC}"
echo -e "${YELLOW}========================================${NC}"
python src/train.py \
    --data_dir "$DATA_DIR" \
    --model_type srm_only \
    --output_dir outputs/exp3_srm_only \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --patience $PATIENCE

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Experiment 3 completed successfully${NC}"
else
    echo -e "${RED}Experiment 3 failed${NC}"
fi

# Experiment 4: Simple Fusion (No DCMA)
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Experiment 4: Simple Fusion${NC}"
echo -e "${YELLOW}========================================${NC}"
python src/train.py \
    --data_dir "$DATA_DIR" \
    --model_type simple_fusion \
    --output_dir outputs/exp4_simple_fusion \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --patience $PATIENCE

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Experiment 4 completed successfully${NC}"
else
    echo -e "${RED}Experiment 4 failed${NC}"
fi

# Experiment 5: Sum Fusion
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Experiment 5: Sum Fusion${NC}"
echo -e "${YELLOW}========================================${NC}"
python src/train.py \
    --data_dir "$DATA_DIR" \
    --model_type sum_fusion \
    --output_dir outputs/exp5_sum_fusion \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --patience $PATIENCE

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Experiment 5 completed successfully${NC}"
else
    echo -e "${RED}Experiment 5 failed${NC}"
fi

echo ""
echo "=========================================="
echo "All Experiments Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check results in outputs/exp*/results.json"
echo "2. Generate visualizations:"
echo "   python src/visualize_ablation.py --model_dir outputs --test_dir $DATA_DIR/test --output_dir visualizations --num_images 10"
echo ""

