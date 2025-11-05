"""
Evaluation Metrics for Face Forgery Detection

Provides utility functions to compute accuracy, precision, recall,
and F1-score, as well as to collect predictions from a model.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred, y_scores=None):
    """
    Calculate accuracy, precision, recall, F1 score, and AUC.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Probability scores for AUC calculation
        
    Returns:
        Dictionary containing all metrics
    """
    # Convert to numpy arrays if they're tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_scores is not None and isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Calculate AUC if scores provided
    if y_scores is not None:
        try:
            auc = roc_auc_score(y_true, y_scores)
            metrics_dict['auc'] = auc
        except ValueError:
            metrics_dict['auc'] = 0.0
    
    return metrics_dict

def get_predictions(model, dataloader, device):
    """
    Generate predictions, scores, and ground truth labels from a DataLoader.

    Args:
        model (torch.nn.Module): Trained model.
        dataloader (DataLoader): DataLoader containing dataset.
        device (torch.device): Device to perform inference on.

    Returns:
        tuple: (predictions, true_labels, scores) as NumPy arrays.
    """
    
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs, _, _ = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
    
    return np.array(all_preds), np.array(all_labels), np.array(all_scores) 