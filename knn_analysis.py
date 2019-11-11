import matplotlib

matplotlib.use('Agg')

import os
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
import torch.nn as nn
import torch
import numpy as np
import scipy.spatial
from sklearn.neighbors import NearestNeighbors
from shutil import copyfile

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--log_dir", required=True, default='log', help="path to log into")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    log_dir = os.path.join(opt.log_dir, 'embeddings')
    all_embeddings = []
    mapping = {}
    count = 0
    for file in os.listdir(log_dir):
        if os.path.isfile(os.path.join(log_dir, file)):
            all_embeddings.append(torch.load(os.path.join(log_dir, file)).data[0].cpu().numpy())
            mapping[count] = file.split('.')[0]
            count += 1

    all_embeddings = np.array(all_embeddings)
    embeddings = np.reshape(all_embeddings, (all_embeddings.shape[0], -1))
    neigh_obj = NearestNeighbors(n_neighbors=5).fit(embeddings)
    distances, neigh = neigh_obj.kneighbors(embeddings)

    try:
        os.mkdir(os.path.join(log_dir, 'neighbours'))
    except FileExistsError:
        pass

    for i in range(0, len(neigh)):
        try:
            os.mkdir(os.path.join(log_dir, 'neighbours', mapping[i]))
            for j in range(0, len(neigh[i])):
                    copyfile(os.path.join(config['dataset_params']['root_dir'], 'all', mapping[i]+".jpg"),
                             os.path.join(log_dir, 'neighbours', mapping[i], mapping[neigh[i][j]]+".jpg"))
        except FileExistsError:
            pass
    # print("saved to neighbors.txt")
    # np.savetxt(log_dir+"/neighbors.txt", neigh, fmt="%d", delimiter=', ')
