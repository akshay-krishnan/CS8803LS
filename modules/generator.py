import torch
from torch import nn
import torch.nn.functional as F

from modules.util import Encoder, Decoder, ResBlock3D
from modules.dense_motion_module import DenseMotionModule, IdentityDeformation
from modules.movement_embedding import MovementEmbeddingModule


class MotionEmbeddingDecoder(nn.Module):
    """
    MLP Encoder
    """
    def __init__(self, in_features, out_features):
        super(MotionEmbeddingDecoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, 2*in_features)
        self.mlp = nn.Sequential(
                  nn.Linear(in_features*8, 4*in_features),
                  nn.Sigmoid(),
                  nn.Linear(4*in_features,out_features*2),
                  nn.Sigmoid(),
                  nn.Linear(out_features*2,out_features),
                  nn.Sigmoid(),
                )

    def forward(self, x):
        x = self.linear(x)
        x = torch.cat([x, x**2, x.sin(), x.cos()], -1)
        return self.mlp(x)

class MotionEmbeddingEncoder(nn.Module):
    """
    MLP Encoder
    """

    def __init__(self, in_features=6, out_features=1):
        super(MotionEmbeddingEncoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, 2*in_features)
        self.mlp = nn.Sequential(
                  nn.Linear(in_features*8, 4*in_features),
                  nn.Sigmoid(),
                  nn.Linear(4*in_features,out_features*2),
                  nn.Sigmoid(),
                  nn.Linear(out_features*2,out_features),
                  nn.Sigmoid(),
                )

    def forward(self, x):
        x = self.linear(x)
        x = torch.cat([x, x**2, x.sin(), x.cos()], -1)
        return self.mlp(x)

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

    def __init__(self, num_channels, num_kp, kp_variance, block_expansion, max_features, num_blocks, embedsize=2):

        super(MotionEmbeddingGenerator, self).__init__()

        self.embedsize = embedsize

        self.motion_encoder = MotionEmbeddingEncoder(in_features=num_kp*12, out_features=embedsize)
        self.motion_decoder = MotionEmbeddingDecoder(in_features=embedsize + num_kp*6, out_features=num_kp*6)
        self.num_kp = num_kp
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        self.motion_encoder.apply(init_weights)
        self.motion_decoder.apply(init_weights)

    def forward(self, d, kp_video, mode=0):
        # mode = 0 (video_prediction)
        # mode = 1 (motion_embed)
        bz,frame,kp,_ = kp_video['mean'].shape

        # kp_video['mean']: [bz, #frame, #kp, 2]
        # kp_video['var']: [bz, #frame, #kp, 2, 2]
        # encode_in: [bz, #frame, #kp*12]
        encode_in = torch.cat([kp_video['mean'].view(bz,frame,-1)[:,:-1,:],
                               kp_video['var'].view(bz,frame,-1)[:,:-1,:],
                               kp_video['mean'].view(bz,frame,-1)[:,1:,:],
                               kp_video['var'].view(bz,frame,-1)[:,1:,:],
                                ], -1)

        # motion_embed: [bz, #frame, 2]
        motion_embed = self.motion_encoder(encode_in)

        if mode == 0:
            decode_in = torch.cat([kp_video['mean'].view(bz,frame,-1)[:,:-1,:],
                                   kp_video['var'].view(bz,frame,-1)[:,:-1,:],
                                   motion_embed
                                    ], -1)
            kp_prediction = self.motion_decoder(decode_in)
            # video_prediction: [bz, ch, #frames, H, W]
            return kp_prediction
        if mode == 1:
            return motion_embed
