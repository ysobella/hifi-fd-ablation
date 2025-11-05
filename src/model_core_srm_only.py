"""
SRM-Only Stream Model for Ablation Study

This is an ablation variant that uses only the frequency (SRM) stream.
"""

import torch
import torch.nn as nn

from components.srm_conv import SRMConv2d_simple, SRMConv2d_Separate
from networks.xception import TransferModel


class SRM_Only_Stream_Net(nn.Module):
    """
    SRM-only single stream model (Ablation Experiment 3).
    
    This variant processes only frequency domain features.
    
    Architecture: SRM filtering -> Xception -> Classifier
    """
    
    def __init__(self):
        super(SRM_Only_Stream_Net, self).__init__()
        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.xception_srm = TransferModel(
            'xception', dropout=0.5, inc=3, return_fea=True)
        self.relu = nn.ReLU(inplace=True)

    def features(self, x):
        """
        Extract features from SRM stream only.
        
        Args:
            x (Tensor): Input image tensor (B, 3, H, W).

        Returns:
            Tensor: SRM feature representation.
        """
        
        srm = self.srm_conv0(x)
        
        x = self.xception_srm.model.fea_part1_0(srm)
        x = self.xception_srm.model.fea_part1_1(x)
        x = self.xception_srm.model.fea_part2(x)
        x = self.xception_srm.model.fea_part3(x)
        x = self.xception_srm.model.fea_part4(x)
        x = self.xception_srm.model.fea_part5(x)
        
        return x

    def classifier(self, fea):
        """
        Apply classifier head to features.

        Args:
            fea (Tensor): Input feature tensor.

        Returns:
            Tuple: (output logits, feature representation)
        """
        
        out, fea = self.xception_srm.classifier(fea)
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
    model = SRM_Only_Stream_Net()
    dummy = torch.rand((1, 3, 256, 256))
    out = model(dummy)
    print("SRM-Only Stream Model")
    print(f"Output shape: {out[0].shape}")

