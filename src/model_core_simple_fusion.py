"""
Simple Fusion Two-Stream Model for Ablation Study

This variant removes dual cross-modal attention and uses simple concatenation fusion.
"""

import torch
import torch.nn as nn

from components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
from networks.xception import TransferModel


class Simple_Fusion_Two_Stream_Net(nn.Module):
    """
    Two-stream model with simple concatenation fusion (Ablation Experiment 4).
    
    This variant removes:
    - Dual cross-modal attention (DCMA)
    - SRM-guided spatial attention
    
    Architecture: RGB + SRM streams -> Concat -> Classifier
    """
    
    def __init__(self):
        super(Simple_Fusion_Two_Stream_Net, self).__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.xception_srm = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        
        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(32, 32)
        self.srm_conv2 = SRMConv2d_Separate(64, 64)
        self.relu = nn.ReLU(inplace=False)  # Changed to False to avoid gradient computation issues
        
        # Simple fusion: concatenation only
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(2048 * 2, 2048, 1, 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )

    def features(self, x):
        """
        Extract multi-modal features with simple fusion.
        
        Args:
            x (Tensor): Input image tensor (B, 3, H, W).

        Returns:
            Tensor: Fused feature representation.
        """
        
        srm = self.srm_conv0(x)
        
        x = self.xception_rgb.model.fea_part1_0(x)
        # Clone x to avoid in-place operation issues when x is used in multiple paths
        y = self.xception_srm.model.fea_part1_0(srm) + self.srm_conv1(x.clone())
        y = self.relu(y)
        
        x = self.xception_rgb.model.fea_part1_1(x)
        # Clone x to avoid in-place operation issues when x is used in multiple paths
        y = self.xception_srm.model.fea_part1_1(y) + self.srm_conv2(x.clone())
        y = self.relu(y)
        
        x = self.xception_rgb.model.fea_part2(x)
        y = self.xception_srm.model.fea_part2(y)
        
        x = self.xception_rgb.model.fea_part3(x)
        y = self.xception_srm.model.fea_part3(y)
        
        x = self.xception_rgb.model.fea_part4(x)
        y = self.xception_srm.model.fea_part4(y)
        
        x = self.xception_rgb.model.fea_part5(x)
        y = self.xception_srm.model.fea_part5(y)
        
        # Simple concatenation fusion
        # Clone to avoid in-place operation issues from Xception blocks
        fea = torch.cat([x.clone(), y.clone()], dim=1)
        fea = self.fusion_conv(fea)
        
        return fea

    def classifier(self, fea):
        """
        Apply classifier head to fused features.

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
    model = Simple_Fusion_Two_Stream_Net()
    dummy = torch.rand((1, 3, 256, 256))
    out = model(dummy)
    print("Simple Fusion Two-Stream Model")
    print(f"Output shape: {out[0].shape}")

