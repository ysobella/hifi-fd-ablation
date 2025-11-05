"""
Hifi-FD Cross-Scale Consistency Visualization Script

This script visualizes broken cross-scale consistency in the Hifi-FD model by analyzing
multi-scale SRM features, cross-scale correlations, and DCMA attention maps. It's designed
to demonstrate how Gaussian blur disrupts cross-scale feature alignment.

Key Visualizations:
1. Multi-Scale SRM Feature Maps: Shows high-frequency components at different scales
2. Cross-Scale Correlation Heatmaps: Computes correlations between SRM outputs across scales
3. DCMA Attention Maps: Visualizes attention weights from Dual Cross-Modality Attention
4. Classification Results: Shows TP, TN, FP, FN samples with confidence scores

Usage:
    python test_visualize_crossscale.py \
        --model_path path/to/model.pth \
        --test_dir path/to/test_directory \
        --output_dir crossscale_visualizations \
        --num_samples 3

The test directory should contain:
    test_dir/
    ├── real/
    │   └── [real images]
    └── fake/
        └── [fake images]
"""

import os
import torch
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import random

from model_core import Two_Stream_Net

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize cross-scale consistency in Hifi-FD model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained Hifi-FD model')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory containing test images (with real/fake subdirectories)')
    parser.add_argument('--output_dir', type=str, default='crossscale_visualizations', help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples per category (TP, TN, FP, FN)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling')
    return parser.parse_args()

def load_image(image_path):
    """
    Load and preprocess a single image using Hifi-FD's normalization.
    
    Args:
        image_path (str): Path to the image.
        
    Returns:
        Tensor: Preprocessed image tensor with batch dimension.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def denormalize_image(image_tensor):
    """
    Denormalize image tensor back to [0, 1] range for visualization.
    
    Args:
        image_tensor (Tensor): Normalized image tensor.
        
    Returns:
        Tensor: Denormalized image tensor.
    """
    # Hifi-FD uses mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(image_tensor.device)
    
    image_tensor = image_tensor * std + mean
    image_tensor = torch.clamp(image_tensor, 0, 1)
    return image_tensor

class CrossScaleAnalyzer:
    """
    Analyzer class for extracting and visualizing cross-scale consistency features.
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Hook storage for intermediate features
        self.srm_features = {}
        self.rgb_features = {}
        self.attention_maps = {}
        
        # Register hooks to capture intermediate features
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        
        def srm_hook(name):
            def hook(module, input, output):
                self.srm_features[name] = output.detach()
            return hook
        
        def rgb_hook(name):
            def hook(module, input, output):
                self.rgb_features[name] = output.detach()
            return hook
        
        # Hook SRM features at different scales
        self.model.srm_conv0.register_forward_hook(srm_hook('srm_scale0'))
        self.model.srm_conv1.register_forward_hook(srm_hook('srm_scale1'))
        self.model.srm_conv2.register_forward_hook(srm_hook('srm_scale2'))
        
        # Hook RGB features at corresponding scales - hook the actual modules used by fea_part methods
        self.model.xception_rgb.model.conv1.register_forward_hook(rgb_hook('rgb_scale0'))
        self.model.xception_rgb.model.conv2.register_forward_hook(rgb_hook('rgb_scale1'))
        self.model.xception_rgb.model.block3.register_forward_hook(rgb_hook('rgb_scale2'))
    
    def analyze_image(self, image_tensor):
        """
        Analyze an image and extract cross-scale consistency features.
        
        Args:
            image_tensor (Tensor): Input image tensor.
            
        Returns:
            dict: Analysis results including predictions and extracted features.
        """
        with torch.no_grad():
            # Clear previous features
            self.srm_features = {}
            self.rgb_features = {}
            
            # Forward pass
            output, features, attention_map = self.model(image_tensor)
            _, predicted = torch.max(output, 1)
            probability = torch.softmax(output, dim=1)
            
            # Debug: print captured features
            print(f"Captured SRM features: {list(self.srm_features.keys())}")
            print(f"Captured RGB features: {list(self.rgb_features.keys())}")
            
            return {
                'prediction': predicted.item(),
                'confidence': probability[0][predicted].item(),
                'attention_map': attention_map,
                'srm_features': self.srm_features.copy(),
                'rgb_features': self.rgb_features.copy(),
                'image_tensor': image_tensor
            }

def visualize_srm_features(srm_features, title_prefix, output_path):
    """
    Visualize multi-scale SRM feature maps.
    
    Args:
        srm_features (dict): Dictionary containing SRM features at different scales
        title_prefix (str): Prefix for the plot title
        output_path (Path): Path to save the visualization
    """
    scales = ['srm_scale0', 'srm_scale1', 'srm_scale2']
    scale_names = ['Scale 0 (Initial)', 'Scale 1 (Mid)', 'Scale 2 (High)']
    
    fig, axes = plt.subplots(len(scales), 4, figsize=(16, 12))
    fig.suptitle(f'{title_prefix} - Multi-Scale SRM Features\nHigh-Frequency Components Across Different Scales', 
                 fontsize=14, fontweight='bold')
    
    for scale_idx, (scale_key, scale_name) in enumerate(zip(scales, scale_names)):
        if scale_key in srm_features:
            srm_tensor = srm_features[scale_key].squeeze(0)  # Remove batch dimension
            srm_np = srm_tensor.cpu().numpy()
            
            # Show first 4 channels
            for ch_idx in range(min(4, srm_np.shape[0])):
                axes[scale_idx, ch_idx].imshow(srm_np[ch_idx], cmap='gray')
                axes[scale_idx, ch_idx].set_title(f'{scale_name}\nChannel {ch_idx}')
                axes[scale_idx, ch_idx].axis('off')
        else:
            # Hide unused subplots
            for ch_idx in range(4):
                axes[scale_idx, ch_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def compute_cross_scale_correlation(srm_features):
    """
    Compute cross-scale correlation matrix for SRM features.
    
    Args:
        srm_features (dict): Dictionary containing SRM features at different scales
        
    Returns:
        np.ndarray: Correlation matrix between scales
    """
    scales = ['srm_scale0', 'srm_scale1', 'srm_scale2']
    scale_names = ['Scale 0', 'Scale 1', 'Scale 2']
    
    correlations = np.zeros((len(scales), len(scales)))
    
    for i, scale1 in enumerate(scales):
        for j, scale2 in enumerate(scales):
            if scale1 in srm_features and scale2 in srm_features:
                # Flatten features for correlation computation
                feat1 = srm_features[scale1].squeeze().cpu().numpy().flatten()
                feat2 = srm_features[scale2].squeeze().cpu().numpy().flatten()
                
                # Compute cosine similarity
                if len(feat1) > 0 and len(feat2) > 0:
                    # Resize to same length if needed
                    min_len = min(len(feat1), len(feat2))
                    feat1 = feat1[:min_len]
                    feat2 = feat2[:min_len]
                    
                    correlation = cosine_similarity([feat1], [feat2])[0][0]
                    correlations[i, j] = correlation
    
    return correlations, scale_names

def visualize_cross_scale_correlation(correlations, scale_names, title_prefix, output_path):
    """
    Visualize cross-scale correlation heatmap.
    
    Args:
        correlations (np.ndarray): Correlation matrix
        scale_names (list): Names of the scales
        title_prefix (str): Prefix for the plot title
        output_path (Path): Path to save the visualization
    """
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(correlations, 
                xticklabels=scale_names, 
                yticklabels=scale_names,
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                vmin=-1, vmax=1,
                fmt='.3f')
    
    plt.title(f'{title_prefix} - Cross-Scale Correlation Matrix\n'
              f'Correlation Between SRM Features Across Different Scales', 
              fontsize=12, fontweight='bold')
    plt.xlabel('SRM Scale')
    plt.ylabel('SRM Scale')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def visualize_attention_map(attention_map, title_prefix, output_path):
    """
    Visualize DCMA attention map.
    
    Args:
        attention_map (Tensor): Attention map tensor
        title_prefix (str): Prefix for the plot title
        output_path (Path): Path to save the visualization
    """
    if attention_map is None:
        return
    
    plt.figure(figsize=(8, 6))
    
    # Convert to numpy and squeeze
    att_np = attention_map.squeeze().cpu().numpy()
    
    # Create visualization
    plt.imshow(att_np, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    plt.title(f'{title_prefix} - DCMA Attention Map\n'
              f'Spatial Attention from Dual Cross-Modality Attention', 
              fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def create_comprehensive_visualization(analysis_result, true_label, category, sample_idx, output_dir):
    """
    Create a comprehensive multi-panel visualization for a single sample.
    
    Args:
        analysis_result (dict): Results from CrossScaleAnalyzer
        true_label (int): Ground truth label
        category (str): Classification category (TP, TN, FP, FN)
        sample_idx (int): Sample index
        output_dir (Path): Output directory
    """
    pred_label = analysis_result['prediction']
    confidence = analysis_result['confidence']
    
    # Create filename prefix
    filename_prefix = f"{category}_sample_{sample_idx:02d}_true_{true_label}_pred_{pred_label}_conf_{confidence:.3f}"
    
    # 1. Visualize input image
    input_img = denormalize_image(analysis_result['image_tensor'].squeeze(0))
    input_pil = transforms.ToPILImage()(input_img)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(input_pil)
    plt.title(f'{category} Sample {sample_idx:02d} - Input Image\n'
              f'True: {true_label}, Pred: {pred_label}, Conf: {confidence:.3f}', 
              fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename_prefix}_input.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    # 2. Visualize multi-scale SRM features
    visualize_srm_features(
        analysis_result['srm_features'], 
        f'{category} Sample {sample_idx:02d}',
        output_dir / f"{filename_prefix}_srm_features.png"
    )
    
    # 3. Compute and visualize cross-scale correlation
    correlations, scale_names = compute_cross_scale_correlation(analysis_result['srm_features'])
    visualize_cross_scale_correlation(
        correlations, scale_names,
        f'{category} Sample {sample_idx:02d}',
        output_dir / f"{filename_prefix}_crossscale_correlation.png"
    )
    
    # 4. Visualize attention map
    visualize_attention_map(
        analysis_result['attention_map'],
        f'{category} Sample {sample_idx:02d}',
        output_dir / f"{filename_prefix}_attention_map.png"
    )
    
    print(f"Saved visualizations for {category} sample {sample_idx}: {filename_prefix}")

def main():
    args = parse_args()
    
    # Set random seed for reproducible sampling
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = Two_Stream_Net().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model from: {args.model_path}")
    
    # Initialize analyzer
    analyzer = CrossScaleAnalyzer(model, device)
    
    # Get test directories
    real_dir = os.path.join(args.test_dir, 'real')
    fake_dir = os.path.join(args.test_dir, 'fake')
    
    # Collect all image paths
    all_images = []
    
    # Add real images
    for img_name in os.listdir(real_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(real_dir, img_name)
            all_images.append((img_path, 0))  # 0 for real
    
    # Add fake images
    for img_name in os.listdir(fake_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(fake_dir, img_name)
            all_images.append((img_path, 1))  # 1 for fake
    
    # Shuffle for random sampling
    random.shuffle(all_images)
    
    # Initialize counters for each category
    categories = ['TP', 'TN', 'FP', 'FN']
    category_counts = {category: 0 for category in categories}
    
    print(f"\nAnalyzing cross-scale consistency with {args.num_samples} samples per category...")
    print("Categories: TP (True Positive), TN (True Negative), FP (False Positive), FN (False Negative)")
    print("Focus: Multi-scale SRM features, cross-scale correlations, and DCMA attention maps")
    
    processed_count = 0
    for img_path, true_label in all_images:
        # Load and analyze image
        image_tensor = load_image(img_path).to(device)
        analysis_result = analyzer.analyze_image(image_tensor)
        
        pred_label = analysis_result['prediction']
        
        # Determine category
        if true_label == 1 and pred_label == 1:
            category = 'TP'
        elif true_label == 0 and pred_label == 0:
            category = 'TN'
        elif true_label == 0 and pred_label == 1:
            category = 'FP'
        elif true_label == 1 and pred_label == 0:
            category = 'FN'
        else:
            continue  # Shouldn't happen
        
        # Create visualizations if we haven't reached the limit for this category
        if category_counts[category] < args.num_samples:
            sample_idx = category_counts[category] + 1
            
            # Create comprehensive visualization
            create_comprehensive_visualization(
                analysis_result, true_label, category, sample_idx, output_dir
            )
            
            category_counts[category] += 1
        
        processed_count += 1
        
        # Check if we have enough samples from all categories
        if all(count >= args.num_samples for count in category_counts.values()):
            print(f"\nCollected {args.num_samples} samples from all categories!")
            print(f"Processed {processed_count} samples out of {len(all_images)} total samples")
            break
    
    # Print final summary
    print(f"\nFinal counts:")
    for category, count in category_counts.items():
        print(f"{category}: {count}/{args.num_samples}")
    
    print(f"\nCross-scale consistency visualizations saved to: {output_dir}")
    print("\nVisualization Components:")
    print("1. Input Image: Original test image (may be blurred)")
    print("2. SRM Features: Multi-scale high-frequency components")
    print("3. Cross-Scale Correlation: Correlation matrix between SRM scales")
    print("4. Attention Map: DCMA spatial attention weights")
    
    print("\nCross-Scale Consistency Analysis:")
    print("- High correlations between SRM scales indicate intact cross-scale consistency")
    print("- Low correlations suggest disrupted multi-resolution dependencies")
    print("- Fragmented attention maps indicate weakened cross-modality attention")
    print("- These visualizations demonstrate how blur disrupts cross-scale feature alignment")

if __name__ == "__main__":
    main()
