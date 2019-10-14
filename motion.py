import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from logger import Logger, Visualizer
from modules.losses import reconstruction_loss
import numpy as np
import imageio
from torch.optim.lr_scheduler import MultiStepLR
from sync_batchnorm import DataParallelWithCallback
from modules.losses import motion_embedding_reconstruction_loss, generator_loss_names



# def generate(generator, appearance_image, kp_appearance, kp_video):
#     out = {'video_prediction': [], 'video_deformed': []}
#     for i in range(kp_video['mean'].shape[1]):
#         kp_target = {k: v[:, i:(i + 1)] for k, v in kp_video.items()}
#         kp_dict_part = {'kp_driving': kp_target, 'kp_source': kp_appearance}
#         out_part = generator(appearance_image, **kp_dict_part)
#         out['video_prediction'].append(out_part['video_prediction'])
#         out['video_deformed'].append(out_part['video_deformed'])

#     out['video_prediction'] = torch.cat(out['video_prediction'], dim=2)
#     out['video_deformed'] = torch.cat(out['video_deformed'], dim=2)
#     out['kp_driving'] = kp_video
#     out['kp_source'] = kp_appearance
#     return out

class MotionGeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, train_params):
        super(MotionGeneratorFullModel, self).__init__()
        self.generator = generator
        self.train_params = train_params

    def forward(self, d, kp_video, x_video):
        
        # video_prediction: [bz, ch, #frames, H, W]
        # x_video: [bz, ch, #frames, H, W]

        video_prediction = self.generator(d, kp_video, x_video)
        loss = motion_embedding_reconstruction_loss(video_prediction, x_video)
        return loss


def train_motion_embedding(config, generator, motion_generator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    
    png_dir = os.path.join(log_dir, 'train_motion_embedding/png')
    log_dir = os.path.join(log_dir, 'train_motion_embedding')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    
    train_params = config['train_motion_embedding_params']
    optimizer_generator = torch.optim.Adam(motion_generator.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))


    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("kp_detector_checkpoint should be specified for mode='test'.")

    start_epoch = 0
    it = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    loss_list = []
    motion_generator_full = MotionGeneratorFullModel(motion_generator, train_params)
    motion_generator_full_par = DataParallelWithCallback(motion_generator_full, device_ids=device_ids)
    
    kp_detector = DataParallelWithCallback(kp_detector)
    generator.eval()
    kp_detector.eval()
    cat_dict = lambda l, dim: {k: torch.cat([v[k] for v in l], dim=dim) for k in l[0]}

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], **train_params['log_params']) as logger:
        for epoch in range(start_epoch, train_params['num_epochs']):
            for it, x in tqdm(enumerate(dataloader)):
                
                with torch.no_grad():
                    
                    # import ipdb; ipdb.set_trace()

                    # x['video']: [bz, ch, #frames, H, W]
                    # detect keypoint for first frame
                    kp_appearance = kp_detector(x['video'][:, :, :1]) 
                    # kp_appearance['mean']: [bz, frame idx, #kp, 2]
                    # kp_appearance['var']: [bz, frame idx, #kp, 2, 2]

                    d = x['video'].shape[2]
                    # kp_video['mean']: [bz, #frame, #kp, 2]
                    # kp_video['var']: [bz, #frame, #kp, 2, 2]
                    kp_video = cat_dict([kp_detector(x['video'][:, :, i:(i + 1)]) for i in range(d)], dim=1)
                    

                loss = motion_generator_full_par(d, kp_video, x['video'])

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                generator_loss_values = [loss.detach().cpu().numpy()]

                logger.log_iter(it,
                                names=generator_loss_names(train_params['loss_weights']),
                                values=generator_loss_values, inp=x)
                it += 1

            scheduler_generator.step()
            logger.log_epoch(epoch, {'generator': generator,
                                     'optimizer_generator': optimizer_generator})