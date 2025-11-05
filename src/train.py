"""
Training Script for Face Forgery Detection with Ablation Study Support

This script trains various ablation variants of the Two-Stream Neural Network.
It supports training:
- Full HiFi (two-stream with all components)
- RGB-only stream
- SRM-only stream
- Simple fusion (no DCMA)
- Sum fusion (concatenation + SE)

For each experiment, it tracks: accuracy, AUC, precision, recall, F1-score, loss
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np

from data_loader import get_data_loaders
from metrics import calculate_metrics, get_predictions

class EarlyStopping:
    """
    Early stopping utility to halt training when validation loss stops improving.

    Args:
        patience (int): Number of epochs to wait after last improvement.
        min_delta (float): Minimum change to qualify as an improvement.
    """
    
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        """
        Check whether training should stop based on validation loss.

        Args:
            val_loss (float): Current validation loss.
        """
        
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def get_model(model_type):
    """
    Factory function to create the appropriate model variant.
    
    Args:
        model_type (str): Type of model to create.
        
    Returns:
        nn.Module: Model instance.
    """
    
    if model_type == 'full':
        from model_core import Two_Stream_Net
        return Two_Stream_Net()
    
    elif model_type == 'rgb_only':
        from model_core_rgb_only import RGB_Only_Stream_Net
        return RGB_Only_Stream_Net()
    
    elif model_type == 'srm_only':
        from model_core_srm_only import SRM_Only_Stream_Net
        return SRM_Only_Stream_Net()
    
    elif model_type == 'simple_fusion':
        from model_core_simple_fusion import Simple_Fusion_Two_Stream_Net
        return Simple_Fusion_Two_Stream_Net()
    
    elif model_type == 'sum_fusion':
        from model_core_sum_fusion import Sum_Fusion_Two_Stream_Net
        return Sum_Fusion_Two_Stream_Net()
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Returns:
        dict: Training metrics for this epoch.
    """
    
    model.train()
    train_loss = 0
    all_preds = []
    all_labels = []
    all_scores = []
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs, _, _ = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        probs = torch.softmax(outputs, dim=1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(probs[:, 1].detach().cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate training metrics
    train_metrics = calculate_metrics(
        np.array(all_labels), 
        np.array(all_preds), 
        np.array(all_scores)
    )
    train_metrics['loss'] = train_loss / len(train_loader)
    
    return train_metrics


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on a dataset.
    
    Returns:
        dict: Evaluation metrics.
    """
    
    model.eval()
    total_loss = 0
    
    all_preds, all_labels, all_scores = get_predictions(model, dataloader, device)
    
    # Calculate loss
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs, _, _ = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_scores)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def train(args):
    """
    Train the model with early stopping and evaluate on test set.
    
    Args:
        args (argparse.Namespace): Parsed training arguments.
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    
    # Create model
    model = get_model(args.model_type).to(device)
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # Training history
    history = {
        'train': [],
        'val': []
    }
    
    # Training loop
    best_val_f1 = 0
    for epoch in range(args.epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'{"="*60}')
        
        # Training phase
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train'].append(train_metrics)
        
        print(f'\nTraining metrics:')
        for metric, value in train_metrics.items():
            print(f'  {metric}: {value:.4f}')
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, criterion, device)
        history['val'].append(val_metrics)
        
        print(f'\nValidation metrics:')
        for metric, value in val_metrics.items():
            print(f'  {metric}: {value:.4f}')
        
        # Early stopping check
        early_stopping(val_metrics['loss'])
        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break
        
        # Save best model
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f'\nSaved best model (F1: {best_val_f1:.4f})')
    
    # Test phase
    print(f'\n{"="*60}')
    print('Testing best model on test set...')
    print(f'{"="*60}')
    
    # Load best model
    model_path = os.path.join(args.output_dir, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, criterion, device)
    history['test'] = test_metrics
    
    print(f'\nTest metrics:')
    for metric, value in test_metrics.items():
        print(f'  {metric}: {value:.4f}')
    
    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\nResults saved to {results_path}')
    print(f'Model saved to {model_path}')


def main():
    """
    Parse command-line arguments and start training.
    """
    
    parser = argparse.ArgumentParser(description='Train and test face forgery detection model')
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing train/val/test subdirectories')
    parser.add_argument('--model_type', type=str, default='full',
                      choices=['full', 'rgb_only', 'srm_only', 'simple_fusion', 'sum_fusion'],
                      help='Model ablation variant to train')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save model and results (auto-generated if None)')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                      help='Number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=5,
                      help='Patience for early stopping')
    
    args = parser.parse_args()
    
    # Auto-generate output directory based on model type
    if args.output_dir is None:
        args.output_dir = os.path.join('outputs', args.model_type)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    train(args)


if __name__ == '__main__':
    main()
