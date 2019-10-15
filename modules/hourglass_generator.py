import torch
from torch import nn
import torch.nn.functional as F

from modules.util import Encoder, Decoder


class MotionGenerator(nn.Module):
    """
    Motion transfer generator. That Given an action embedding and an 
    appearance trying to reconstruct the target frame.
    """

    def __init__(self, num_channels, block_expansion, max_features, num_blocks):
        super(MotionTransferGenerator, self).__init__()

        self.appearance_encoder = Encoder(block_expansion, in_features=num_channels, 
                max_features=max_features, num_blocks=num_blocks)

        self.video_decoder = Decoder(block_expansion=block_expansion, in_features=num_channels, 
                out_features=num_channels, max_features=max_features, num_blocks=num_blocks, 
                additional_features_for_block=embedding_features, use_last_conv=False)

    def forward(self, source_image):
        appearance_skips = self.appearance_encoder(source_image)
        video_prediction = self.video_decoder(appearance_skips)
        video_prediction = torch.sigmoid(video_prediction)

        return video_prediction
