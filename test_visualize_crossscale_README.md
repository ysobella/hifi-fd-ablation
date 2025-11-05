# Hifi-FD Cross-Scale Consistency Visualization

This document describes the `test_visualize_crossscale.py` script, which provides comprehensive visualizations of cross-scale consistency in the Hifi-FD model. The script is specifically designed to demonstrate how Gaussian blur disrupts cross-scale feature alignment.

## Overview

The Hifi-FD model relies on multi-scale high-frequency feature extraction and cross-modality attention mechanisms. When cross-scale dependencies are disrupted (e.g., by Gaussian blurring), the model's performance deteriorates significantly. This script visualizes these disruptions through multiple analysis components.

## Key Features

### 1. Multi-Scale SRM Feature Visualization
- **Purpose**: Shows high-frequency components extracted at different scales
- **Implementation**: Captures SRM features from `srm_conv0`, `srm_conv1`, and `srm_conv2`
- **Visualization**: Displays first 4 channels from each scale in a grid layout
- **Interpretation**: 
  - Clear, distinct patterns indicate intact cross-scale consistency
  - Blurred or degraded patterns suggest disrupted frequency relationships

### 2. Cross-Scale Correlation Analysis
- **Purpose**: Computes correlations between SRM features across different scales
- **Implementation**: Uses cosine similarity between flattened SRM feature maps
- **Visualization**: Heatmap showing correlation matrix between scales
- **Interpretation**:
  - High correlations (close to 1) indicate strong cross-scale consistency
  - Low correlations suggest broken multi-resolution dependencies

### 3. DCMA Attention Map Visualization
- **Purpose**: Shows spatial attention weights from Dual Cross-Modality Attention
- **Implementation**: Extracts attention maps from the model's forward pass
- **Visualization**: Heatmap of attention weights overlaid on spatial dimensions
- **Interpretation**:
  - Focused attention patterns indicate effective cross-modality correlation
  - Fragmented or weak attention suggests disrupted cross-scale relationships

### 4. Classification Category Analysis
- **TP (True Positive)**: Correctly identified fake images
- **TN (True Negative)**: Correctly identified real images  
- **FP (False Positive)**: Incorrectly identified real images as fake
- **FN (False Negative)**: Incorrectly identified fake images as real

## Script Architecture

### CrossScaleAnalyzer Class
The core analysis is performed by the `CrossScaleAnalyzer` class, which:

1. **Registers Forward Hooks**: Captures intermediate features during model forward pass
2. **Extracts Multi-Scale Features**: Collects SRM and RGB features at different scales
3. **Performs Analysis**: Computes correlations and attention patterns
4. **Generates Visualizations**: Creates comprehensive multi-panel figures

### Hook Registration
```python
# Hook SRM features at different scales
self.model.srm_conv0.register_forward_hook(srm_hook('srm_scale0'))
self.model.srm_conv1.register_forward_hook(srm_hook('srm_scale1'))
self.model.srm_conv2.register_forward_hook(srm_hook('srm_scale2'))

# Hook RGB features at corresponding scales
self.model.xception_rgb.model.fea_part1_0.register_forward_hook(rgb_hook('rgb_scale0'))
self.model.xception_rgb.model.fea_part1_1.register_forward_hook(rgb_hook('rgb_scale1'))
self.model.xception_rgb.model.fea_part2.register_forward_hook(rgb_hook('rgb_scale2'))
```

## Usage

### Basic Usage
```bash
python test_visualize_crossscale.py \
    --model_path path/to/trained_model.pth \
    --test_dir path/to/test_directory \
    --output_dir crossscale_visualizations \
    --num_samples 3
```

### Advanced Usage
```bash
python test_visualize_crossscale.py \
    --model_path path/to/trained_model.pth \
    --test_dir path/to/blurred_test_directory \
    --output_dir blur_analysis \
    --num_samples 5 \
    --seed 123
```

### Parameters
- `--model_path`: Path to trained Hifi-FD model (.pth file)
- `--test_dir`: Directory containing test images with `real/` and `fake/` subdirectories
- `--output_dir`: Directory to save visualizations (default: `crossscale_visualizations`)
- `--num_samples`: Number of samples per category (default: 3)
- `--seed`: Random seed for reproducible sampling (default: 42)

## Input Structure
```
test_dir/
├── real/
│   └── [real images - may be blurred]
└── fake/
    └── [fake images - may be blurred]
```

## Output Structure
```
output_dir/
├── TP_sample_01_true_1_pred_1_conf_0.856_input.png
├── TP_sample_01_true_1_pred_1_conf_0.856_srm_features.png
├── TP_sample_01_true_1_pred_1_conf_0.856_crossscale_correlation.png
├── TP_sample_01_true_1_pred_1_conf_0.856_attention_map.png
├── TN_sample_01_true_0_pred_0_conf_0.923_input.png
├── TN_sample_01_true_0_pred_0_conf_0.923_srm_features.png
├── TN_sample_01_true_0_pred_0_conf_0.923_crossscale_correlation.png
├── TN_sample_01_true_0_pred_0_conf_0.923_attention_map.png
└── ... (similar files for FP and FN categories)
```

## File Naming Convention
```
{category}_sample_{index:02d}_true_{true_label}_pred_{pred_label}_conf_{confidence:.3f}_{visualization_type}.png
```

Where:
- `category`: TP, TN, FP, or FN
- `index`: Sample number (01, 02, etc.)
- `true_label`: Ground truth label (0=real, 1=fake)
- `pred_label`: Predicted label (0=real, 1=fake)
- `confidence`: Model confidence score
- `visualization_type`: input, srm_features, crossscale_correlation, or attention_map

## Visualization Components

### 1. Input Image
- **File**: `*_input.png`
- **Content**: Original test image (denormalized)
- **Purpose**: Shows the input that was analyzed
- **Note**: May be blurred if using pre-blurred datasets

### 2. Multi-Scale SRM Features
- **File**: `*_srm_features.png`
- **Content**: 3x4 grid showing SRM features at different scales
- **Layout**: 
  - Rows: Different scales (Scale 0, Scale 1, Scale 2)
  - Columns: First 4 channels from each scale
- **Purpose**: Demonstrates high-frequency component extraction

### 3. Cross-Scale Correlation
- **File**: `*_crossscale_correlation.png`
- **Content**: Heatmap showing correlations between SRM scales
- **Color Scale**: Red (high correlation) to Blue (low correlation)
- **Purpose**: Shows consistency between different scales

### 4. Attention Map
- **File**: `*_attention_map.png`
- **Content**: Spatial attention weights from DCMA
- **Color Scale**: Hot colormap (red=high attention, blue=low attention)
- **Purpose**: Shows where the model focuses attention

## Cross-Scale Consistency Analysis

### Intact Cross-Scale Consistency (Low Blur)
- **SRM Features**: Clear, distinct high-frequency patterns across scales
- **Correlations**: High correlation values (0.7-1.0) between scales
- **Attention**: Focused attention patterns highlighting relevant regions
- **Performance**: High confidence predictions, correct classifications

### Disrupted Cross-Scale Consistency (High Blur)
- **SRM Features**: Blurred or degraded patterns, loss of fine details
- **Correlations**: Lower correlation values (0.3-0.7) between scales
- **Attention**: Fragmented or weak attention patterns
- **Performance**: Lower confidence, increased misclassifications

### Expected Patterns by Category

#### TP (True Positive) - Correctly Detected Fakes
- Clear high-frequency artifacts in SRM features
- Strong cross-scale correlations
- Focused attention on manipulated regions
- High confidence scores

#### TN (True Negative) - Correctly Detected Reals
- Natural frequency patterns without artificial artifacts
- Moderate cross-scale correlations
- Attention focused on natural features
- High confidence scores

#### FP (False Positive) - Real Images Misclassified as Fake
- Natural textures that resemble forgery artifacts
- Moderate cross-scale correlations
- Attention may focus on natural high-frequency patterns
- Lower confidence scores

#### FN (False Negative) - Fake Images Misclassified as Real
- Degraded or inconsistent frequency patterns
- Low cross-scale correlations
- Weak or fragmented attention patterns
- Low confidence scores

## Integration with Blur Analysis

This script is designed to work with pre-blurred datasets to demonstrate progressive degradation:

### Low Blur (Kernel Size 5)
- Minimal impact on cross-scale consistency
- High correlations between SRM scales
- Clear attention patterns
- Maintained classification performance

### Medium Blur (Kernel Size 10-15)
- Moderate degradation of frequency relationships
- Reduced correlations between scales
- Some fragmentation in attention patterns
- Slight performance degradation

### High Blur (Kernel Size 20-25)
- Significant disruption of cross-scale dependencies
- Low correlations between SRM scales
- Fragmented attention patterns
- Substantial performance degradation

## Technical Implementation Details

### Feature Extraction
- Uses PyTorch forward hooks to capture intermediate features
- Extracts features at multiple scales during single forward pass
- Preserves spatial dimensions for correlation analysis

### Correlation Computation
- Flattens feature maps for correlation computation
- Uses cosine similarity for robust correlation measurement
- Handles different feature map sizes gracefully

### Visualization Generation
- Uses matplotlib and seaborn for high-quality visualizations
- Consistent color schemes across all visualizations
- High-resolution output (150 DPI) for publication quality

### Memory Management
- Clears feature storage between samples
- Uses `torch.no_grad()` for inference efficiency
- Efficient tensor operations to minimize memory usage

## Dependencies
- PyTorch
- torchvision
- matplotlib
- seaborn
- numpy
- PIL (Pillow)
- scikit-learn
- pathlib

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Missing Features**: Ensure model has all required components
3. **Empty Visualizations**: Check that test directory has images
4. **Correlation Errors**: Verify feature extraction hooks are working

### Debug Mode
Add debug prints to verify feature extraction:
```python
print(f"SRM features keys: {list(self.srm_features.keys())}")
print(f"RGB features keys: {list(self.rgb_features.keys())}")
```

## Related Work

This script supports the analysis described in the thesis:

> "Frequency models performed slightly lower than spatial, but still higher than the models with attention mechanisms. Despite having a stronger initial performance (F1-score of 71.74% for FreqNet and 74.94% for Hifi-FD), both models deteriorate more rapidly at higher blur levels, likely due to broken cross-scale consistency."

The visualizations provide concrete evidence of how cross-scale consistency breaks down under different degradation conditions, supporting the theoretical analysis of frequency model robustness.
