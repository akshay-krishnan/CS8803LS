import matplotlib

matplotlib.use('Agg')

import os
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset
from modules.content_encoder import ContentEncoder
from modules.motion_encoder import Encoder
from convolutional_rnn import Conv2dGRU
from modules.frame_decoder import FrameDecoder
from augmentation import ContentMotionTestTransform

from train_update1 import train
from reconstruction import reconstruction
from transfer import transfer
from prediction import prediction
from infer_update1 import infer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "knn", "infer", "transfer"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'],
                            transform=ContentMotionTestTransform(**config['dataset_params']['augmentation_params']),
                            is_knn=(opt.mode == 'knn' or opt.mode == 'transfer'))

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' video_gen ' + strftime("%d-%m-%y %H:%M:%S", gmtime())

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    content_encoder = ContentEncoder(**config['model_params']['content_encoder_params'])
    content_encoder.to(opt.device_ids[0])
    if opt.verbose:
        print("Content encoder architecture \n", content_encoder)

    motion_encoder = Encoder(**config['model_params']['motion_encoder_params'])
    motion_encoder.to(opt.device_ids[0])
    if opt.verbose:
        print("Motion encoder architecture, \n", motion_encoder)

    sequence_model = Conv2dGRU(256, 256, 3, num_layers=2)
    sequence_model.to(opt.device_ids[0])
    if opt.verbose:
        print("sequence model architecture, \n", sequence_model)

    decoder = FrameDecoder(in_features=512, num_blocks=5)
    decoder.to(opt.device_ids[0])
    if opt.verbose:
        print("decoder model architecture, \n", decoder)

    if opt.mode == 'train':
        print("Training...")
        train(config, content_encoder, motion_encoder, sequence_model, decoder,
              opt.checkpoint, log_dir, dataset, opt.device_ids, is_video_test_split=True)
    elif opt.mode == 'transfer':
        print("Transfer...")
        transfer(config, content_encoder, motion_encoder, sequence_model, decoder,
              opt.checkpoint, log_dir, dataset, opt.device_ids)
    elif opt.mode == "infer":
        print("Inference...")
        infer(config, content_encoder, motion_encoder, sequence_model, decoder,
              opt.checkpoint, log_dir, dataset, opt.device_ids, is_video_test_split=True)
    elif opt.mode == "knn":
        print("Analysis...")
        infer(config, content_encoder, motion_encoder, sequence_model, decoder,
              opt.checkpoint, log_dir, dataset, opt.device_ids, is_video_test_split=True,
              is_analysis=True)