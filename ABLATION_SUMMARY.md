# HiFi-FD Ablation Study Summary

## Quick Overview

This ablation study evaluates 5 variants of the HiFi-FD architecture to understand the contribution of different components.

## Ablation Variants

| Exp | Variant Name | Streams | DCMA | SRM-Att | Fusion Method |
|-----|-------------|---------|------|---------|---------------|
| 1 | Full HiFi | RGB+SRM | ✓ | ✓ | Concat+SE (default) |
| 2 | RGB-Only | RGB only | ✗ | ✗ | N/A |
| 3 | SRM-Only | SRM only | ✗ | ✗ | N/A |
| 4 | Simple Fusion | RGB+SRM | ✗ | ✗ | Concat (simple) |
| 5 | Sum Fusion | RGB+SRM | ✓ | ✓ | Sum (additive) |

## Key Questions

1. **Q1: Does two-stream help?**
   - Compare: Full HiFi (Exp 1) vs RGB-only (Exp 2) vs SRM-only (Exp 3)
   - Expected: Full HiFi > both single streams

2. **Q2: Does DCMA matter?**
   - Compare: Full HiFi (Exp 1) vs Simple Fusion (Exp 4)
   - Expected: Full HiFi ≥ Simple Fusion (if DCMA adds value)

3. **Q3: Does fusion method matter?**
   - Compare: Full HiFi (Exp 1) vs Sum Fusion (Exp 5)
   - Expected: Full HiFi ≥ Sum Fusion (if concat+SE is better)

## Architecture Differences

### Exp 1: Full HiFi (Baseline)
```python
# Two streams
RGB features + SRM features
↓
Dual Cross-Modal Attention (DCMA)  # RGB⟷SRM interaction
↓
SRM-guided spatial attention on RGB
↓
Concatenation + SE attention fusion
↓
Classifier
```

### Exp 2: RGB-Only
```python
# Single stream
RGB features only
↓
No DCMA, no SRM attention
↓
Classifier
```

### Exp 3: SRM-Only
```python
# Single stream  
SRM features only
↓
No DCMA, no RGB processing
↓
Classifier
```

### Exp 4: Simple Fusion
```python
# Two streams
RGB features + SRM features
↓
No DCMA (streams run independently)
↓
Simple concatenation fusion
↓
Classifier
```

### Exp 5: Sum Fusion
```python
# Two streams
RGB features + SRM features
↓
Dual Cross-Modal Attention (DCMA)
↓
SRM-guided spatial attention
↓
Sum fusion (x + y, no SE)
↓
Classifier
```

## Running the Experiments

### Quick Start
```bash
# Update the data path in the script
vim run_ablation_experiments.sh

# Run all experiments
./run_ablation_experiments.sh
```

### Individual Runs
```bash
# Full HiFi
python src/train.py --data_dir /path/to/data --model_type full

# RGB-Only
python src/train.py --data_dir /path/to/data --model_type rgb_only

# SRM-Only
python src/train.py --data_dir /path/to/data --model_type srm_only

# Simple Fusion
python src/train.py --data_dir /path/to/data --model_type simple_fusion

# Sum Fusion
python src/train.py --data_dir /path/to/data --model_type sum_fusion
```

## Expected Results Table

| Metric | Full | RGB-Only | SRM-Only | Simple Fusion | Sum Fusion |
|--------|------|----------|----------|---------------|------------|
| Accuracy | - | Lower | Lower | Similar/Lower | Similar |
| F1-Score | - | Lower | Lower | Similar/Lower | Similar |
| AUC | - | Lower | Lower | Similar/Lower | Similar |

## Visualizations

After training, generate comparison images:

```bash
python src/visualize_ablation.py \
    --model_dir outputs \
    --test_dir /path/to/test \
    --output_dir visualizations \
    --num_images 10
```

This creates:
- 10 side-by-side comparison images
- Attention map overlays (if available)
- Confidence scores for each variant
- Summary text file

## File Structure

```
hifi-fd-ablation/
├── src/
│   ├── model_core.py              # Full HiFi (baseline)
│   ├── model_core_rgb_only.py     # Exp 2: RGB-only
│   ├── model_core_srm_only.py     # Exp 3: SRM-only
│   ├── model_core_simple_fusion.py # Exp 4: Simple fusion
│   ├── model_core_sum_fusion.py   # Exp 5: Sum fusion
│   ├── train.py                    # Training script
│   ├── visualize_ablation.py       # Visualization script
│   ├── metrics.py                  # Metrics calculation
│   └── data_loader.py             # Data loading
├── outputs/
│   ├── exp1_full/
│   ├── exp2_rgb_only/
│   ├── exp3_srm_only/
│   ├── exp4_simple_fusion/
│   └── exp5_sum_fusion/
├── visualizations/                 # Generated visualizations
├── ABLATION_README.md             # Full documentation
├── ABLATION_SUMMARY.md            # This file
└── run_ablation_experiments.sh    # Runner script
```

## Interpretation Guide

### If Full > RGB-only and Full > SRM-only:
- **Conclusion**: Two-stream architecture is beneficial
- **Intuition**: Both RGB and SRM streams contribute unique information

### If Full ≈ Simple Fusion:
- **Conclusion**: DCMA may not be critical
- **Intuition**: Simple concatenation sufficient for fusion

### If Full > Simple Fusion:
- **Conclusion**: DCMA provides value
- **Intuition**: Cross-modal attention improves feature learning

### If Full ≈ Sum Fusion:
- **Conclusion**: Fusion method matters little
- **Intuition**: Both concatenation and sum are effective

### If Full > Sum Fusion:
- **Conclusion**: Concatenation+SE fusion is better
- **Intuition**: Channel attention improves feature selection

## Contact

For questions, issues, or contributions, please refer to the main repository.

