###############################################################
# For inquiries regarding this code, please contact:          #
#               anjialin8@gmail.com                           #
#                                                             #
# In this study, the dataset is divided into two parts:       #
# training and testing. Please contact the author to obtain   #
# the dataset.                                                #
###############################################################

import torch
from thop import profile
from AttMamba import AttMamba
import torch.nn.functional as F
from fb import LightweightPatchEmbed

class Swish(torch.nn.Module):
    """Swish activation function module."""
    def forward(self, x):
        return x * torch.sigmoid(x)

class SMNet(torch.nn.Module):
    """
    SMNet: A lightweight neural network model for anomaly detection
    in industrial applications, utilizing sparse convolution and
    attention mechanisms.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels1 (int): Number of output channels for the first convolutional layer.
        out_channels2 (int): Number of output channels for the sparse layer.
        kernel_size1 (int): Kernel size for the first convolutional layer.
        kernel_size2 (int): Kernel size for the sparse convolution layer.
        padding (int): Padding size for convolutional layers.
        stride (int): Stride for the first convolutional layer.
        linear (int): Input size for the final linear layer.
    """
    
    def __init__(self, in_channels, out_channels1, out_channels2, kernel_size1, kernel_size2, padding, stride, linear) -> None:
        super(SMNet, self).__init__()

        self.LightweightPatchEmbed = LightweightPatchEmbed(in_channels=3, patch_size=16, emb_dim=64)

        self.conv1 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_channels),
            torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels1, kernel_size=kernel_size1, padding=padding, stride=stride),
            Swish()
        )

        self.mamba = AttMamba(
            d_model=out_channels2,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(linear, 2)

    def forward(self, input):
        input = self.LightweightPatchEmbed(input)
        x = self.conv1(input)
        x = torch.transpose(x, 1, 2)
        x = self.mamba(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return self.linear(x)

# Model initialization and profiling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SMNet(in_channels=257, out_channels1=128, out_channels2=128, kernel_size1=5, kernel_size2=3, padding=2, stride=1, linear=64).to(device)

# Generate a sample input tensor for profiling
sample_input = torch.randn(1, 3, 256, 256).to(device)
flops, params = profile(model, inputs=(sample_input,))
print(f'MFLOPs: {flops / 1e6:.2f} M')
print(f'Parameters: {params / 1e3:.2f} K')

# Model inference
output = model(sample_input)
print(f"Output Shape: {output.shape}")
