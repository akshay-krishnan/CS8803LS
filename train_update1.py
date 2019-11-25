from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from modules.losses import video_reconstruction_loss
from torch.optim.lr_scheduler import MultiStepLR
from sync_batchnorm import DataParallelWithCallback

class GeneratorFullModel(torch.nn.Module):

    def __init__(self, content_encoder, motion_encoder, sequence_model, decoder, train_params,
                 is_video_test_split=False):
        super(GeneratorFullModel, self).__init__()
        self.content_encoder = content_encoder
        self.motion_encoder = motion_encoder
        self.sequence_model = sequence_model
        self.decoder = decoder
        self.train_params = train_params
        self.is_video_test_split = is_video_test_split

    def forward(self, x):

        content_embedding = self.content_encoder(x['image'])
        print("content embedding", content_embedding[-1].shape)
        motion_embedding = self.motion_encoder(x['video'])
        print("motion embedding", motion_embedding[-1].shape)
        generated_embedding, h = self.sequence_model(motion_embedding[-1])
        print("generated embedding", generated_embedding.shape, type(generated_embedding))
        generated_video = self.decoder(content_embedding[-1], generated_embedding)
        print("generated video ", generated_video.shape)
        # print("original video size", x['video'].shape)
        if self.is_video_test_split:
            losses = video_reconstruction_loss(x['test'], generated_video)
        else:
            losses = video_reconstruction_loss(x['video'], generated_video)
        return losses, generated_video


def train(config, content_encoder, motion_encoder, sequence_model, decoder, checkpoint, log_dir, dataset, device_ids, is_video_test_split=False):
    train_params = config['train_params']

    optimizer_content = torch.optim.Adam(content_encoder.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    optimizer_motion = torch.optim.Adam(motion_encoder.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    optimizer_sequence = torch.optim.Adam(sequence_model.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch, it = Logger.load_checkpoint(checkpoint, content_encoder, motion_encoder, sequence_model,
                                          decoder, optimizer_content, optimizer_motion, optimizer_sequence,
                                          optimizer_decoder)
    else:
        start_epoch = 0
        it = 0

    scheduler_content = MultiStepLR(optimizer_content, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_motion = MultiStepLR(optimizer_motion, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_sequence = MultiStepLR(optimizer_sequence, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=start_epoch - 1)
    scheduler_decoder = MultiStepLR(optimizer_decoder, train_params['epoch_milestones'], gamma=0.1,
                                    last_epoch=start_epoch - 1)

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    generator_full = GeneratorFullModel(content_encoder, motion_encoder, sequence_model, decoder, train_params,
                                        is_video_test_split=is_video_test_split)
    generator_full_par = DataParallelWithCallback(generator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], **train_params['log_params']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                out = generator_full_par(x)
                loss = out[-2]
                print("epoch ", epoch, "loss ", loss)
                generated = out[-1]
                loss.backward(retain_graph=not train_params['detach_kp_discriminator'])
                optimizer_content.step()
                optimizer_content.zero_grad()
                optimizer_motion.step()
                optimizer_motion.zero_grad()
                optimizer_sequence.step()
                optimizer_sequence.zero_grad()
                optimizer_decoder.step()
                optimizer_decoder.zero_grad()

                logger.log_each_iteration(it, loss, inp=x, out=generated)
                it += 1

            scheduler_content.step()
            scheduler_motion.step()
            scheduler_sequence.step()
            scheduler_decoder.step()

            logger.log_epoch(epoch, {'content_encoder': content_encoder,
                                     'motion_encoder': motion_encoder,
                                     'sequence_model': sequence_model,
                                     'decoder': decoder,
                                     'optimizer_content': optimizer_content,
                                     'optimizer_motion': optimizer_motion,
                                     'optimizer_sequence': optimizer_sequence,
                                     'optimizer_decoder': optimizer_decoder})