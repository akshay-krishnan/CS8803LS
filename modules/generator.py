import torch
from torch import nn
import torch.nn.functional as F

from modules.util import Encoder, Decoder, ResBlock3D, MotionEmbeddingEncoder, MotionEmbeddingDecoder
from modules.dense_motion_module import DenseMotionModule, IdentityDeformation
from modules.movement_embedding import MovementEmbeddingModule


class MotionTransferGenerator(nn.Module):
    """
    Motion transfer generator. That Given a keypoints and an appearance trying to reconstruct the target frame.
    Produce 2 versions of target frame, one warped with predicted optical flow and other refined.
    """

    def __init__(self, num_channels, num_kp, kp_variance, block_expansion, max_features, num_blocks, num_refinement_blocks,
                 dense_motion_params=None, kp_embedding_params=None, interpolation_mode='nearest'):
        super(MotionTransferGenerator, self).__init__()

        self.appearance_encoder = Encoder(block_expansion, in_features=num_channels, max_features=max_features,
                                          num_blocks=num_blocks)

        if kp_embedding_params is not None:
            self.kp_embedding_module = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance,
                                                               num_channels=num_channels, **kp_embedding_params)
            embedding_features = self.kp_embedding_module.out_channels
        else:
            self.kp_embedding_module = None
            embedding_features = 0

        if dense_motion_params is not None:
            self.dense_motion_module = DenseMotionModule(num_kp=num_kp, kp_variance=kp_variance,
                                                         num_channels=num_channels,
                                                         **dense_motion_params)
        else:
            self.dense_motion_module = IdentityDeformation()

        self.video_decoder = Decoder(block_expansion=block_expansion, in_features=num_channels,
                                     out_features=num_channels, max_features=max_features, num_blocks=num_blocks,
                                     additional_features_for_block=embedding_features,
                                     use_last_conv=False)

        self.refinement_module = torch.nn.Sequential()
        in_features = block_expansion + num_channels + embedding_features
        for i in range(num_refinement_blocks):
            self.refinement_module.add_module('r' + str(i),
                                              ResBlock3D(in_features, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        self.refinement_module.add_module('conv-last', nn.Conv3d(in_features, num_channels, kernel_size=1, padding=0))
        self.interpolation_mode = interpolation_mode

    def deform_input(self, inp, deformations_absolute):
        bs, d, h_old, w_old, _ = deformations_absolute.shape
        _, _, _, h, w = inp.shape
        deformations_absolute = deformations_absolute.permute(0, 4, 1, 2, 3)
        deformation = F.interpolate(deformations_absolute, size=(d, h, w), mode=self.interpolation_mode)
        deformation = deformation.permute(0, 2, 3, 4, 1)
        deformed_inp = F.grid_sample(inp, deformation)
        return deformed_inp

    def forward(self, source_image, kp_driving, kp_source):
        appearance_skips = self.appearance_encoder(source_image)

        deformations_absolute = self.dense_motion_module(source_image=source_image, kp_driving=kp_driving,
                                                         kp_source=kp_source)

        deformed_skips = [self.deform_input(skip, deformations_absolute) for skip in appearance_skips]

        if self.kp_embedding_module is not None:
            d = kp_driving['mean'].shape[1]
            movement_embedding = self.kp_embedding_module(source_image=source_image, kp_driving=kp_driving,
                                                          kp_source=kp_source)
            kp_skips = [F.interpolate(movement_embedding, size=(d,) + skip.shape[3:], mode=self.interpolation_mode) for skip in appearance_skips]
            skips = [torch.cat([a, b], dim=1) for a, b in zip(deformed_skips, kp_skips)]
        else:
            skips = deformed_skips

        video_deformed = self.deform_input(source_image, deformations_absolute)
        video_prediction = self.video_decoder(skips)
        video_prediction = self.refinement_module(video_prediction)
        video_prediction = torch.sigmoid(video_prediction)

        return {"video_prediction": video_prediction, "video_deformed": video_deformed}


class MotionEmbeddingGenerator(nn.Module):
    """
    Motion embedding generator.
    """

    def __init__(self, num_channels, num_kp, kp_variance, block_expansion, max_features, num_blocks):

        super(MotionEmbeddingGenerator, self).__init__()

        self.motion_encoder = MotionEmbeddingEncoder(in_features=6, out_features=1, num_blocks=num_blocks)
        self.motion_decoder = MotionEmbeddingDecoder(block_expansion=block_expansion, in_features=num_channels + num_kp,
                                                     out_features=num_channels, max_features=max_features, num_blocks=num_blocks,
                                                     use_last_conv=False)
        self.num_kp = num_kp

    def forward(self, d, kp_video, x_video):


        # kp_video['mean']: [bz, #frame, #kp, 2]
        # kp_video['var']: [bz, #frame, #kp, 2, 2]
        # x_video: [bz, ch, #frames, H, W]
        video_prediction = []
        kp_diff = {}
        for i in range(d - 1):
            
            kp_diff['mean'] = kp_video['mean'][:, i + 1] - kp_video['mean'][:, i]
            kp_diff['var'] = kp_video['var'][:, i + 1] - kp_video['var'][:, i]
            
            kp_diff['mean'] = torch.squeeze(kp_diff['mean'], 1)
            kp_diff['var'] = kp_diff['var'].view(kp_diff['var'].shape[0], self.num_kp, -1)

            source_diff = torch.cat((kp_diff['mean'], kp_diff['var']), 2)
            source_image = torch.squeeze(x_video[:, :, i], 2)

            # source_diff: [bz, #kp, 6]
            # motion_embed: [bz, #kp, 1] -> [bz, #kp, H, W]
            motion_embed = self.motion_encoder(x = source_diff)
            motion_embed = motion_embed.unsqueeze(3).repeat(1, 1, x_video.shape[-2], x_video.shape[-1])

            # source_image: [bz, ch, H, W]
            # motion_embed: [bz, #kp, H, W]
            prediction = self.motion_decoder(source_image = source_image, motion_embed = motion_embed)

            # prediction: [bz, ch, H, W] -> [bz, ch, 1, H, W]
            video_prediction.append(prediction)

        # video_prediction: [bz, ch, #frames, H, W]
        video_prediction = torch.stack(video_prediction, 2)
        return video_prediction