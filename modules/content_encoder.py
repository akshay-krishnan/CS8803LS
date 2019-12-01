from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d

class ContentEncoder(nn.Module):
    def __init__(self, block_expansion, in_features=3, num_blocks=3, max_features=256):
        super(ContentEncoder, self).__init__()
        down_blocks = []
        kernel_size = 3
        padding = 1
        self.block_expansion = block_expansion
        self.num_blocks = num_blocks
        self.max_features = max_features
        for i in range(num_blocks):
            down_blocks.append(DownBlock2D(in_features if i == 0 else min(max_features, block_expansion * (2 ** (i-1))),
                                           min(max_features, block_expansion * (2 ** i)),
                                           kernel_size=kernel_size, padding=padding))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]

        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs

    def getOutputSize(self, frames, height, width):
        o_height = (1.0*height)/(2**self.num_blocks)
        o_width = (1.0*width)/(2**self.num_blocks)
        o_frames = min(min(self.max_features, self.block_expansion * (2 ** (self.num_blocks))))
        return (o_frames, o_height, o_width)


class DownBlock2D(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(DownBlock2D, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, stride=1) # change stride if using pooling
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out