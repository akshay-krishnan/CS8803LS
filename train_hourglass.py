import matplotlib

matplotlib.use('Agg')

import os
import yaml
import torch 
from torch.utils.data import DataLoader
from tqdm import trange

from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
from logger import Logger
from modules.losses import reconstruction_loss, generator_loss_names
from torch.optim.lr_scheduler import MultiStepLR
from sync_batchnorm import DataParallelWithCallback
from frames_dataset import FramesDataset
from modules.util import Encoder, Decoder, Hourglass


class HourglassFullModel(torch.nn.Module):

    def __init__(self, hourglass):
        super(HourglassFullModel, self).__init__()
        self.hourglass = hourglass

    def forward(self, x): 
        generated = self.hourglass(x)
        loss = reconstruction_loss(generated, x, 1.0)
        return (loss, generated)


def load_cpk(checkpoint_path, content_hourglass, optimizer_content_hourglass):
    checkpoint = torch.load(checkpoint_path)
    content_hourglass.load_state_dict(checkpoint["content_hourglass"])
    optimizer.load_state_dict(checkpoint["optimizer_content_hourglass"])
    return checkpoint['epoch'], checkpoint['it']


def train(config, content_hourglass, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']
    optimizer_content_hourglass = torch.optim.Adam(content_hourglass.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch, it = load_cpk(checkpoint, content_hourglass, optimizer_content_hourglass)
    else:
        start_epoch  = 0
        it = 0
    scheduler_content_hourglass = MultiStepLR(optimizer_content_hourglass, train_params['epoch_milestones'], gamma=0.1, last_epoch = start_epoch -1)
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4, drop_last = True)

    hourglass_full = HourglassFullModel(content_hourglass)
    hourglass_full_par = DataParallelWithCallback(hourglass_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], **train_params['log_params']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                losses, generated = hourglass_full_par(x['source'])
                losses = [val.mean() for val in losses]
                loss = sum(losses)
                loss.backward()
                optimizer_content_hourglass.step()
                optimizer_content_hourglass.zero_grad()

                losses = [val.detach().cpu().numpy() for val in losses]
                logger.log_iter_hourglass(it, names='content_hourglass', 
                        values=losses, inp=x, out=generated)
                it+=1
            scheduler_content_hourglass.step()
            logger.log_epoch(epoch, {'content_hourglass': content_hourglass})



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="path to log directory")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))), help="Names of the devices comma separated.")
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--config", required=True, help="path to config")

    opt = parser.parse_args()
    config = yaml.load(open(opt.config))
    
    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d-%m-%y %H:%M:%S", gmtime())
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)
    

    content_hourglass = Hourglass(**config["model_params"]["hourglass_params"])
    content_hourglass.to(opt.device_ids[0])
    print(content_hourglass)

    dataset = FramesDataset(is_train=True, **config['dataset_params'])
    
    train(config, content_hourglass, opt.checkpoint, log_dir, dataset, opt.device_ids)
    
