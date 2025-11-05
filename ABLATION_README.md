# HiFi-FD Ablation Study

This document provides instructions for running the ablation study experiments on the HiFi-FD (High-Frequency Features for Face Forgery Detection) model.

## Overview

This ablation study evaluates the contribution of different architectural components in the HiFi-FD two-stream neural network for face forgery detection. The study compares five model variants:

1. **Full HiFi** - Complete two-stream architecture with all components
2. **RGB-Only Stream** - Single-stream using only spatial (RGB) features
3. **SRM-Only Stream** - Single-stream using only frequency (SRM) features
4. **Simple Fusion** - Two-stream with simple concatenation (no cross-modal attention)
5. **Sum Fusion** - Two-stream with additive fusion (concatenation + SE attention)

## Architecture Components

### Full HiFi (Baseline)
- ✓ Two streams (RGB + SRM)
- ✓ Dual cross-modal attention (DCMA)
- ✓ SRM-guided spatial attention
- ✓ Concatenation + SE fusion

### RGB-Only Stream
- ✓ RGB stream only
- ✗ Removed SRM stream
- ✗ Removed DCMA
- ✗ Removed SRM attention

### SRM-Only Stream
- ✓ SRM stream only
- ✗ Removed RGB stream
- ✗ Removed DCMA
- ✗ Removed RGB attention

### Simple Fusion
- ✓ Two streams (RGB + SRM)
- ✗ Removed DCMA
- ✗ Removed SRM-guided attention
- ✓ Simple concatenation fusion

### Sum Fusion
- ✓ Two streams (RGB + SRM)
- ✓ DCMA
- ✓ SRM-guided attention
- ✓ Additive fusion (x + y) instead of concat + SE

## Installation

```bash
# Install required packages
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset in the following structure:

```
data_dir/
├── train/
│   ├── real/
│   │   └── *.jpg
│   └── fake/
│       └── *.jpg
├── val/
│   ├── real/
│   │   └── *.jpg
│   └── fake/
│       └── *.jpg
└── test/
    ├── real/
    │   └── *.jpg
    └── fake/
        └── *.jpg
```

## Running Experiments

### 1. Train All Ablation Variants

Run each experiment separately:

```bash
# Full HiFi (Baseline)
python src/train.py --data_dir /path/to/data --model_type full --output_dir outputs/exp1_full --batch_size 8 --epochs 100

# RGB-Only Stream
python src/train.py --data_dir /path/to/data --model_type rgb_only --output_dir outputs/exp2_rgb_only --batch_size 8 --epochs 100

# SRM-Only Stream
python src/train.py --data_dir /path/to/data --model_type srm_only --output_dir outputs/exp3_srm_only --batch_size 8 --epochs 100

# Simple Fusion (No DCMA)
python src/train.py --data_dir /path/to/data --model_type simple_fusion --output_dir outputs/exp4_simple_fusion --batch_size 8 --epochs 100

# Sum Fusion
python src/train.py --data_dir /path/to/data --model_type sum_fusion --output_dir outputs/exp5_sum_fusion --batch_size 8 --epochs 100
```

### 2. Quick Start Script

Run all experiments automatically:

```bash
# Create a shell script
cat > run_all_experiments.sh << 'EOF'
#!/bin/bash

DATA_DIR="/path/to/data"
BATCH_SIZE=8
EPOCHS=100

# Full HiFi
python src/train.py --data_dir $DATA_DIR --model_type full \
    --output_dir outputs/exp1_full --batch_size $BATCH_SIZE --epochs $EPOCHS

# RGB-Only
python src/train.py --data_dir $DATA_DIR --model_type rgb_only \
    --output_dir outputs/exp2_rgb_only --batch_size $BATCH_SIZE --epochs $EPOCHS

# SRM-Only
python src/train.py --data_dir $DATA_DIR --model_type srm_only \
    --output_dir outputs/exp3_srm_only --batch_size $BATCH_SIZE --epochs $EPOCHS

# Simple Fusion
python src/train.py --data_dir $DATA_DIR --model_type simple_fusion \
    --output_dir outputs/exp4_simple_fusion --batch_size $BATCH_SIZE --epochs $EPOCHS

# Sum Fusion
python src/train.py --data_dir $DATA_DIR --model_type sum_fusion \
    --output_dir outputs/exp5_sum_fusion --batch_size $BATCH_SIZE --epochs $EPOCHS

echo "All experiments completed!"
EOF

chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

## Metrics Tracked

For each experiment, the following metrics are tracked on train, validation, and test sets:

- **Accuracy**: Percentage of correct predictions
- **AUC**: Area Under the ROC Curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Loss**: Cross-entropy loss

Results are automatically saved to `{output_dir}/results.json` for each experiment.

## Generating Visualizations

After training all models, generate comparison visualizations:

```bash
python src/visualize_ablation.py \
    --model_dir outputs \
    --test_dir /path/to/test \
    --output_dir visualizations \
    --num_images 10
```

This will:
1. Load the same 10 images from your test set
2. Generate predictions using all 5 ablation variants
3. Create side-by-side comparison visualizations
4. Save visualizations to the `visualizations` directory

### Visualization Output

For each image, the script generates:
- A grid showing the original image alongside predictions from all variants
- Attention maps overlayed (if available)
- Confidence scores for each variant
- A summary text file comparing all predictions

## Interpreting Results

### Expected Findings

1. **Full HiFi vs RGB/SRM-Only**: 
   - Should demonstrate that dual-stream > single-stream
   - Both streams contribute valuable information

2. **Full HiFi vs Simple Fusion**:
   - Tests the importance of dual cross-modal attention
   - If Full > Simple Fusion significantly, DCMA is important

3. **Full HiFi vs Sum Fusion**:
   - Tests the importance of fusion method complexity
   - Concatenation + SE attention vs simple addition

### Analysis Questions

- **Q1**: Does the two-stream architecture improve over single streams?
  - Compare: Full, RGB-only, SRM-only test metrics
  - Expected: Full > RGB-only ≈ SRM-only

- **Q2**: Does dual cross-modal attention matter?
  - Compare: Full vs Simple Fusion
  - Expected: Full ≥ Simple Fusion

- **Q3**: Does fusion method matter?
  - Compare: Full vs Sum Fusion
  - Expected: Similar or Full slightly better

## Output Structure

After running all experiments:

```
outputs/
├── exp1_full/
│   ├── best_model.pth
│   └── results.json
├── exp2_rgb_only/
│   ├── best_model.pth
│   └── results.json
├── exp3_srm_only/
│   ├── best_model.pth
│   └── results.json
├── exp4_simple_fusion/
│   ├── best_model.pth
│   └── results.json
└── exp5_sum_fusion/
    ├── best_model.pth
    └── results.json

visualizations/
├── image1_comparison.png
├── image2_comparison.png
├── ...
└── summary_comparison.txt
```

## Example Results

Each `results.json` file contains:

```json
{
  "train": [
    {
      "accuracy": 0.95,
      "auc": 0.98,
      "precision": 0.93,
      "recall": 0.97,
      "f1_score": 0.95,
      "loss": 0.15
    },
    ...
  ],
  "val": [...],
  "test": {
    "accuracy": 0.92,
    "auc": 0.95,
    "precision": 0.90,
    "recall": 0.94,
    "f1_score": 0.92,
    "loss": 0.20
  }
}
```

## Command-Line Arguments

### Training Script

```
--data_dir        : Path to dataset directory (required)
--model_type      : Model variant to train (default: full)
                   Options: full, rgb_only, srm_only, simple_fusion, sum_fusion
--output_dir      : Output directory (auto-generated if not specified)
--batch_size      : Batch size (default: 8)
--num_workers     : Number of data loading workers (default: 8)
--learning_rate   : Learning rate (default: 0.001)
--epochs          : Number of training epochs (default: 100)
--patience        : Early stopping patience (default: 5)
```

### Visualization Script

```
--model_dir       : Directory containing model checkpoints (default: outputs)
--test_dir        : Test directory with real/fake subdirectories (required)
--output_dir      : Directory to save visualizations (default: visualizations)
--num_images      : Number of images to visualize (default: 10)
--device          : Device to run inference on (cpu or cuda)
```

## Troubleshooting

### Issue: Model checkpoint not found
- Make sure all experiments have been trained first
- Check that model paths in `outputs/` match expected structure

### Issue: CUDA out of memory
- Reduce `--batch_size` (e.g., 4 or 2)
- Use `--device cpu` for CPU-only inference

### Issue: Results not reproducible
- Set random seeds for reproducibility:
```python
import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
```

## Citation

If you use this ablation study in your research, please cite the original HiFi-FD paper:

```bibtex
@InProceedings{Luo_2021_CVPR,
    author    = {Luo, Yuchen and Zhang, Yong and Yan, Junchi and Liu, Wei},
    title     = {Generalizing Face Forgery Detection With High-Frequency Features},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {16317-16326}
}
```

## Contact

For questions or issues with this ablation study implementation, please open an issue on the repository.

## License

See LICENSE file for details.

