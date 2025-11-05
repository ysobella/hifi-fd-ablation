"""
RGB-Only Stream Model for Ablation Study

This is an ablation variant that removes the entire SRM stream and uses only 
RGB features from the spatial domain.
"""

import torch
import torch.nn as nn

from networks.xception import TransferModel


class RGB_Only_Stream_Net(nn.Module):
    """
    RGB-only single stream model (Ablation Experiment 2).
    
    This variant removes:
    - SRM stream
    - SRM convolutions
    - Dual cross-modal attention
    - SRM-guided spatial attention
    
    Architecture: Single RGB backbone (Xception) -> Classifier
    """
    
    def __init__(self):
        super(RGB_Only_Stream_Net, self).__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)

    def features(self, x):
        """
        Extract features from RGB stream only.
        
        Args:
            x (Tensor): Input image tensor (B, 3, H, W).

        Returns:
            Tensor: RGB feature representation.
        """
        
        x = self.xception_rgb.model.fea_part1_0(x)
        x = self.xception_rgb.model.fea_part1_1(x)
        x = self.xception_rgb.model.fea_part2(x)
        x = self.xception_rgb.model.fea_part3(x)
        x = self.xception_rgb.model.fea_part4(x)
        x = self.xception_rgb.model.fea_part5(x)
        
        return x

    def classifier(self, fea):
        """
        Apply classifier head to features.

        Args:
            fea (Tensor): Input feature tensor.

        Returns:
            Tuple: (output logits, feature representation)
        """
        
        out, fea = self.xception_rgb.classifier(fea)
        return out, fea

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input image tensor (B, 3, H, W)

        Returns:
            tuple: (logits, features, None)
        """
        
        out, fea = self.classifier(self.features(x))
        return out, fea, None


if __name__ == '__main__':
    model = RGB_Only_Stream_Net()
    dummy = torch.rand((1, 3, 256, 256))
    out = model(dummy)
    print("RGB-Only Stream Model")
    print(f"Output shape: {out[0].shape}")

