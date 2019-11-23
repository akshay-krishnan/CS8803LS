import os

import torch
from torch.utils.data import DataLoader

from frames_dataset import PairedDataset
from logger import Logger
import numpy as np

from sync_batchnorm import DataParallelWithCallback
from infer_update1 import GeneratorFullModel


def transfer(config, content_encoder, motion_encoder, sequence_model, decoder, checkpoint, log_dir,
          dataset, device_ids):
    log_dir = os.path.join(log_dir, 'transfer')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if checkpoint is not None:
        start_epoch, it = Logger.load_checkpoint(checkpoint, content_encoder, motion_encoder, sequence_model,
                                          decoder)
    else:
        print("no checkpoint provided")
        return None

    dataset = PairedDataset(initial_dataset=dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    transfer_params = config['transfer_params']
    generator_full = GeneratorFullModel(content_encoder, motion_encoder, sequence_model, decoder, transfer_params,
                                        is_video_test_split=True, log_dir=log_dir)

    generator_full_par = DataParallelWithCallback(generator_full, device_ids=device_ids)
    total_loss = 0

    with torch.no_grad():
        with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                    **transfer_params['log_params'], is_train=False) as logger:
            for x in dataloader:
                out = generator_full_par(x)
                loss = out[-2]
                generated = out[-1]
                logger.log_each_iteration(it, loss, x, generated)
                total_loss += loss
                it += 1
                print("iter : ", it, " loss: ", loss)

    print("reconstruction loss on test data ", total_loss/len(dataloader.dataset))