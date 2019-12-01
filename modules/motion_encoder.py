from torchvision.models import video
import torch
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


class DownBlock3D(nn.Module):
    """
    Simple block for processing video (encoder).
    """
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(DownBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features=3, num_blocks=4, max_features=128):
        super(Encoder, self).__init__()

        down_blocks = []

        kernel_size = 3
        padding = 1
        for i in range(num_blocks):
            down_blocks.append(DownBlock3D(in_features if i == 0 else min(max_features, block_expansion * (2 ** (i-1))),
                                           min(max_features, block_expansion * (2 **i)),
                                           kernel_size=kernel_size, padding=padding))
            self.out_features = min(max_features, block_expansion * (2 **i))

        # down_blocks.append(nn.Conv3d(in_channels=min(max_features, block_expansion * (2 ** num_blocks)),
        #                              out_channels=1, kernel_size=kernel_size, padding=padding))
        # down_blocks.append(BatchNorm3d(1, affine=True))

        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]

        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        # outs.append(torch.mean(outs[-1], -3))
        outs.append(outs[-1].transpose(-3,-4))
        return outs

    def getOutputFeatures(self):
        return self.out_features

# class MotionEncoder(nn.Module):
#     def __init__(self):
#         super(MotionEncoder, self).__init__()
#
#         model = video.r2plus1d_18(pretrained=True, progress=True)
#         self.model = nn.Sequential(list(model.children()[]))
#
#     def forward(self):