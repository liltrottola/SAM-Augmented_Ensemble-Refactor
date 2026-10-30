import torch
import torch.nn as nn
from SAM2UNet import SAM2UNet



class ChannelReduction(nn.Module):
    def __init__(self, in_channels, residual=False, kernel_size=1):
        super().__init__()
        self.residual = residual
        self.channel_reduction = nn.Sequential(
            # kernel_size only on the first conv (1 = per-pixel; 3 = 3x3 neighbourhood); padding keeps the resolution
            nn.Conv2d(in_channels, 32, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 3, kernel_size=1)
        )
        if residual:
            nn.init.zeros_(self.channel_reduction[-1].weight)
            nn.init.zeros_(self.channel_reduction[-1].bias)

    def forward(self, x):
        out = self.channel_reduction(x)
        if self.residual:
            return x[:,:3] + out
        return out

class SAM2UNet_N3(nn.Module):
    # checkpoint_path is passed to SAM2UNet as in Train_vanilla (Hiera-L from the SAM2 checkpoint at training,
    # None at test: all weights come from the .pth). The encoder stays frozen inside SAM2UNet; the block is trained
    # (its gradient goes through the frozen encoder, whose parameters are not updated)
    def __init__(self, in_channels, residual=False, kernel_size=1, checkpoint_path=None):
        super().__init__()
        self.reduce = ChannelReduction(in_channels , residual, kernel_size)
        self.sam2unet = SAM2UNet(checkpoint_path)

    def forward(self, x):
        x3 = self.reduce(x)
        return self.sam2unet(x3)   # final head + 2 side outputs as SAM2UNet
