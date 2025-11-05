"""
Sum Fusion Two-Stream Model for Ablation Study

This variant uses additive fusion instead of concatenation + SE attention.
"""

import torch
import torch.nn as nn

from components.attention import DualCrossModalAttention
from components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
from model_core import SRMPixelAttention
from networks.xception import TransferModel


class Sum_Fusion_Two_Stream_Net(nn.Module):
    """
    Two-stream model with sum fusion (Ablation Experiment 5).
    
    This variant:
    - Keeps all HiFi components (DCMA, SRM attention)
    - Uses sum fusion (x + y) instead of concat + SE attention
    
    Architecture: RGB + SRM streams with attention -> Sum -> Classifier
    """
    
    def __init__(self):
        super(Sum_Fusion_Two_Stream_Net, self).__init__()
        self.xception_rgb = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.xception_srm = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        
        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(32, 32)
        self.srm_conv2 = SRMConv2d_Separate(64, 64)
        self.relu = nn.ReLU(inplace=True)
        
        self.att_map = None
        self.srm_sa = SRMPixelAttention(3)
        self.srm_sa_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dual_cma0 = DualCrossModalAttention(in_dim=728, ret_att=False)
        self.dual_cma1 = DualCrossModalAttention(in_dim=728, ret_att=False)

    def features(self, x):
        """
        Extract multi-modal features with sum fusion.
        
        Args:
            x (Tensor): Input image tensor (B, 3, H, W).

        Returns:
            Tensor: Fused feature representation.
        """
        
        srm = self.srm_conv0(x)
        
        x = self.xception_rgb.model.fea_part1_0(x)
        y = self.xception_srm.model.fea_part1_0(srm) + self.srm_conv1(x)
        y = self.relu(y)
        
        x = self.xception_rgb.model.fea_part1_1(x)
        y = self.xception_srm.model.fea_part1_1(y) + self.srm_conv2(x)
        y = self.relu(y)
        
        # SRM guided spatial attention
        self.att_map = self.srm_sa(srm)
        x = x * self.att_map + x
        x = self.srm_sa_post(x)
        
        x = self.xception_rgb.model.fea_part2(x)
        y = self.xception_srm.model.fea_part2(y)
        
        x, y = self.dual_cma0(x, y)
        
        x = self.xception_rgb.model.fea_part3(x)
        y = self.xception_srm.model.fea_part3(y)
        
        x, y = self.dual_cma1(x, y)
        
        x = self.xception_rgb.model.fea_part4(x)
        y = self.xception_srm.model.fea_part4(y)
        
        x = self.xception_rgb.model.fea_part5(x)
        y = self.xception_srm.model.fea_part5(y)
        
        # Sum fusion (x + y)
        fea = x + y
        
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
            tuple: (logits, features, attention map)
        """
        
        out, fea = self.classifier(self.features(x))
        return out, fea, self.att_map


if __name__ == '__main__':
    model = Sum_Fusion_Two_Stream_Net()
    dummy = torch.rand((1, 3, 256, 256))
    out = model(dummy)
    print("Sum Fusion Two-Stream Model")
    print(f"Output shape: {out[0].shape}")

