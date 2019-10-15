from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


class FrameDecoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, out_features, num_blocks=3, max_features=256):
        super(FrameDecoder, self).__init__()
        kernel_size = 3
        padding = 1

        up_blocks = []

        for i in range(num_blocks):
            up_blocks.append(UpBlock2D(int(in_features/2**i), int(in_features/2**(i+1)), kernel_size=kernel_size, padding=padding))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.conv1 = nn.Conv2d(in_channels=int(in_features/2**num_blocks),
                                  out_channels=int(in_features/2**(num_blocks+1)), kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=int(in_features/2**(num_blocks+1)),
                                  out_channels=3, kernel_size=kernel_size, padding=padding)
        self.input = torch.Tensor

    def forward(self, content, motion):
        content.unsqueeze_(-4)
        content = content.expand_as(motion)
        self.input = torch.cat((content, motion), dim=-3)
        out = []
        for i in range(self.input.shape[-4]):
            out_time = self.input[:, i]
            for block in self.up_blocks:
                out_time = block(out_time)
            out_time = self.conv1(out_time)
            out_time = self.conv2(out_time)
            out.append(out_time)
        out = [temp.unsqueeze(-3) for temp in out]
        out = torch.cat(out, dim=-3)
        return out


class UpBlock2D(nn.Module):
    """
    Simple block for processing video (decoder).
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super(UpBlock2D, self).__init__()

        # self.conv = nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features,
        #                                kernel_size=kernel_size, padding=padding, stride=2)
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=(2, 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out