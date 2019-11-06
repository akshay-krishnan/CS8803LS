import torch
from torch.utils.data import DataLoader
from logger import Logger
from modules.losses import video_reconstruction_loss
from sync_batchnorm import DataParallelWithCallback

class GeneratorFullModel(torch.nn.Module):

    def __init__(self, content_encoder, motion_encoder, sequence_model, decoder, infer_params,
                 is_video_test_split=False):
        super(GeneratorFullModel, self).__init__()
        self.content_encoder = content_encoder
        self.motion_encoder = motion_encoder
        self.sequence_model = sequence_model
        self.decoder = decoder
        self.infer_params = infer_params
        self.is_video_test_split = is_video_test_split

    def forward(self, x):
        content_embedding = self.content_encoder(x['image'])
        # print("content embedding", content_embedding[-1].shape)
        motion_embedding = self.motion_encoder(x['video'])
        # print("motion embedding", motion_embedding[-1].shape)
        generated_embedding, h = self.sequence_model(motion_embedding[-1])
        # print("generated embedding", generated_embedding.shape, type(generated_embedding))
        generated_video = self.decoder(content_embedding[-1], generated_embedding)
        # print("generated video ", generated_video.shape)
        # print("original video size", x['video'].shape)
        if self.is_video_test_split:
            losses = video_reconstruction_loss(x['test'], generated_video)
        else:
            losses = video_reconstruction_loss(x['video'], generated_video)
        return losses, generated_video


def infer(config, content_encoder, motion_encoder, sequence_model, decoder, checkpoint, log_dir, dataset, device_ids):

    infer_params = config['infer_params']

    if checkpoint is not None:
        start_epoch, it = Logger.load_checkpoint(checkpoint, content_encoder, motion_encoder, sequence_model,
                                          decoder)
    else:
        print("no checkpoint provided")
        return None

    dataloader = DataLoader(dataset, batch_size=infer_params['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    generator_full = GeneratorFullModel(content_encoder, motion_encoder, sequence_model, decoder, infer_params)
    generator_full_par = DataParallelWithCallback(generator_full, device_ids=device_ids)
    total_loss = 0

    with torch.no_grad():
        with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                    **infer_params['log_params'], is_train=False) as logger:
            for x in dataloader:
                out = generator_full_par(x)
                loss = out[-2]
                generated = out[-1]
                total_loss += loss
                logger.log_each_iteration(it, loss, x, generated)
                it += 1
                print("iter : ", it, " loss: ", loss)

    print("reconstruction loss on test data ", total_loss/len(dataloader.dataset))
    print("total loss ", total_loss)

    return