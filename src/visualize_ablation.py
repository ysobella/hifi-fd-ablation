"""
Visualization Script for Ablation Study

This script generates comparison visualizations showing predictions from different
ablation variants on the same set of images.
"""

import os
import torch
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model_core import Two_Stream_Net
from model_core_rgb_only import RGB_Only_Stream_Net
from model_core_srm_only import SRM_Only_Stream_Net
from model_core_simple_fusion import Simple_Fusion_Two_Stream_Net
from model_core_sum_fusion import Sum_Fusion_Two_Stream_Net


def get_model(model_type, checkpoint_path=None):
    """
    Load the appropriate model variant.
    
    Args:
        model_type (str): Type of model.
        checkpoint_path (str): Path to model checkpoint.
        
    Returns:
        nn.Module: Loaded model.
    """
    
    if model_type == 'full':
        model = Two_Stream_Net()
    elif model_type == 'rgb_only':
        model = RGB_Only_Stream_Net()
    elif model_type == 'srm_only':
        model = SRM_Only_Stream_Net()
    elif model_type == 'simple_fusion':
        model = Simple_Fusion_Two_Stream_Net()
    elif model_type == 'sum_fusion':
        model = Sum_Fusion_Two_Stream_Net()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    model.eval()
    return model


def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for inference.
    
    Args:
        image_path (str): Path to image.
        
    Returns:
        tuple: (original_image, preprocessed_tensor)
    """
    
    # Load original image
    original_img = Image.open(image_path).convert('RGB')
    
    # Preprocess for model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    preprocessed = transform(original_img)
    
    return original_img, preprocessed.unsqueeze(0)


def predict_with_model(model, image_tensor, device):
    """
    Get prediction from model.
    
    Args:
        model: Trained model.
        image_tensor: Preprocessed image tensor.
        device: Device to run inference on.
        
    Returns:
        dict: Prediction results.
    """
    
    with torch.no_grad():
        output, _, att_map = model(image_tensor.to(device))
        probs = torch.softmax(output, dim=1)
        _, pred = torch.max(output, 1)
        
        confidence = probs[0][pred].item()
        class_name = 'Fake' if pred.item() == 1 else 'Real'
        
        return {
            'prediction': class_name,
            'confidence': confidence,
            'prob_real': probs[0][0].item(),
            'prob_fake': probs[0][1].item(),
            'attention_map': att_map
        }


def visualize_single_image(image_path, model_dir, output_dir, device):
    """
    Generate visualization for a single image across all ablation variants.
    
    Args:
        image_path (str): Path to image to visualize.
        model_dir (str): Directory containing model checkpoints.
        output_dir (str): Directory to save visualizations.
        device: Device to run inference on.
    """
    
    # Model variants to test
    variants = ['full', 'rgb_only', 'srm_only', 'simple_fusion', 'sum_fusion']
    
    # Load original image
    original_img, preprocessed_tensor = load_and_preprocess_image(image_path)
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Visualization: {os.path.basename(image_path)}', fontsize=16, y=0.98)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Show original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=10)
    axes[0].axis('off')
    
    # Get predictions from all variants
    results = {}
    for i, variant in enumerate(variants):
        checkpoint_path = os.path.join(model_dir, variant, 'best_model.pth')
        
        # Skip if checkpoint doesn't exist
        if not os.path.exists(checkpoint_path):
            print(f"Warning: No checkpoint found for {variant}")
            continue
        
        try:
            # Load model
            model = get_model(variant, checkpoint_path).to(device)
            
            # Get prediction
            result = predict_with_model(model, preprocessed_tensor, device)
            results[variant] = result
            
            # Display image with prediction
            ax = axes[i + 1]
            ax.imshow(original_img)
            
            # Add attention map if available
            if result['attention_map'] is not None:
                att_map = result['attention_map'].squeeze().cpu().numpy()
                if len(att_map.shape) == 2:
                    ax.imshow(att_map, cmap='jet', alpha=0.3)
            
            title = f"{variant}\n{result['prediction']}\nConf: {result['confidence']:.3f}"
            ax.set_title(title, fontsize=8)
            ax.axis('off')
            
        except Exception as e:
            print(f"Error processing {variant}: {e}")
            continue
    
    # Remove unused axes
    for i in range(len(variants) + 1, 9):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = os.path.join(output_dir, f'{os.path.basename(image_path)}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_path}")
    
    return results


def visualize_ablation_study(model_dir, test_dir, output_dir, num_images=10, device='cpu'):
    """
    Generate visualizations for multiple images across all ablation variants.
    
    Args:
        model_dir (str): Directory containing model checkpoints.
        test_dir (str): Test directory with real/fake subdirectories.
        output_dir (str): Directory to save visualizations.
        num_images (int): Number of images to visualize.
        device: Device to run inference on.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect images from test directory
    real_dir = os.path.join(test_dir, 'real')
    fake_dir = os.path.join(test_dir, 'fake')
    
    all_images = []
    
    # Add real images
    if os.path.exists(real_dir):
        for img_name in os.listdir(real_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                all_images.append((os.path.join(real_dir, img_name), 'real'))
    
    # Add fake images
    if os.path.exists(fake_dir):
        for img_name in os.listdir(fake_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                all_images.append((os.path.join(fake_dir, img_name), 'fake'))
    
    # Limit to num_images
    all_images = all_images[:num_images]
    
    # Generate visualizations
    all_results = {}
    for i, (image_path, label) in enumerate(all_images):
        print(f"\nProcessing image {i+1}/{len(all_images)}: {os.path.basename(image_path)}")
        results = visualize_single_image(image_path, model_dir, output_dir, device)
        all_results[os.path.basename(image_path)] = results
    
    # Create summary comparison
    create_summary_comparison(all_results, output_dir)


def create_summary_comparison(results, output_dir):
    """
    Create a summary comparison of all results.
    
    Args:
        results (dict): Dictionary of results for each image.
        output_dir (str): Directory to save summary.
    """
    
    # Create summary table
    summary_lines = []
    summary_lines.append("Image Name | Full | RGB-Only | SRM-Only | Simple Fusion | Sum Fusion")
    summary_lines.append("-" * 80)
    
    for img_name, predictions in results.items():
        line = f"{img_name[:20]}"
        for variant in ['full', 'rgb_only', 'srm_only', 'simple_fusion', 'sum_fusion']:
            if variant in predictions:
                pred = predictions[variant]
                line += f" | {pred['prediction']} ({pred['confidence']:.2f})"
            else:
                line += " | -"
        summary_lines.append(line)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'summary_comparison.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\nSummary saved to {summary_path}")


def main():
    """
    Main entry point for visualization script.
    """
    
    parser = argparse.ArgumentParser(description='Generate ablation study visualizations')
    
    parser.add_argument('--model_dir', type=str, default='outputs',
                      help='Directory containing model checkpoints')
    parser.add_argument('--test_dir', type=str, required=True,
                      help='Test directory with real/fake subdirectories')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                      help='Directory to save visualizations')
    parser.add_argument('--num_images', type=int, default=10,
                      help='Number of images to visualize')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda'],
                      help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"Starting ablation visualization")
    print(f"Model directory: {args.model_dir}")
    print(f"Test directory: {args.test_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of images: {args.num_images}")
    print(f"Device: {device}")
    
    # Generate visualizations
    visualize_ablation_study(
        args.model_dir,
        args.test_dir,
        args.output_dir,
        args.num_images,
        device
    )
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()

