
import ase
import yaml
import time
import copy
import joblib
import pickle
import numpy as np
import datetime
from ase import Atoms, io, build
import warnings
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple
import pandas as pd
from pathlib import Path
import glob, re, os

import os
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

CONFIG = {}

CONFIG['gpu'] = True

CONFIG['data_x_path'] = 'data_x/mp_20_raw_train_dist_mat.pt'
CONFIG['data_ead_path'] = 'data_x/mp_20_raw_train_ead_mat.pt'
CONFIG['composition_path'] = 'data_x/mp_20_raw_train_composition_mat.pt'
CONFIG['cell_path'] = 'data_x/mp_20_raw_train_cell_mat.pt'
CONFIG['data_y_path'] = "data_x/mp_20/raw_train/targets.csv"

CONFIG['unprocessed_path'] = 'data_x/mp_20_raw_train_unprocessed.txt'

CONFIG['model_path1'] = 'saved_models/diff_1d_diffusion_mp.pt'
CONFIG['model_path2'] = 'saved_models/diff_1d_unet_mp.pt'
CONFIG['scaler_path'] = 'saved_models/diff_scaler_mp.gz'


def read_data(data_dir: Union[str, Path] = "data"):

    unprocessed = set()
    with open(CONFIG['unprocessed_path'], 'r') as f:
        for l in f.readlines():
            unprocessed.add(int(l))

    dist_mat = torch.load(CONFIG['data_x_path']).to("cpu")
    ead_mat = torch.load(CONFIG['data_ead_path']).to("cpu")
    composition_mat = torch.load(CONFIG['composition_path']).to("cpu")
    cell_mat = torch.load(CONFIG['cell_path']).to("cpu")

    # build index
    _ind = [i for i in range(dist_mat.shape[0]) if i not in unprocessed]
    indices = torch.tensor(_ind, dtype=torch.long).to("cpu")

    # select rows torch.Size([27136, 1])
    dist_mat = dist_mat[indices] # the torch.load needs the index in tensor format to convert the loaded file in a tensor.
    ead_mat = ead_mat[indices]
    composition_mat = composition_mat[indices]
    cell_mat = cell_mat[indices]

    # normalize composition
    sums = torch.sum(composition_mat, axis=1).view(-1,1)
    composition_mat = composition_mat / sums
    composition_mat = torch.cat((composition_mat, sums), dim=1)

    y = []
    with open(CONFIG['data_y_path'], 'r') as f:
        for i, d in enumerate(f.readlines()):
            if i not in unprocessed:
                y.append(float(d.split(',')[1]))

    data_y = np.reshape(np.array(y), (-1,1))

    data_y = torch.from_numpy(data_y)
    data_y = data_y.to(torch.float32)

    data_x = torch.cat((ead_mat/1000000, dist_mat, cell_mat, composition_mat, data_y), dim=1)

    print(data_x.shape)
    print(data_y.shape)

    mask = data_x[:, 600] <= 10
    data_x = data_x[mask]

    data_x, composition_mat, data_y = data_x[:, 0:607], data_x[:,607:708] , data_x[:,708]

    scaler = MinMaxScaler()
    scaler.fit(data_x)
    data_x = scaler.transform(data_x)
    joblib.dump(scaler, CONFIG['scaler_path']) # save the scaler to be used for later purpose on testing data.

    comp1, comp2 = composition_mat[:, 0:96], composition_mat[:,100]/20
    comp1 = (comp1.to(torch.float32))
    comp2 = comp2.to(torch.float32).view(-1,1)

    composition_mat_add = torch.cat((comp1,comp2), dim=1)

    data_x = torch.from_numpy(data_x)
    data_x = data_x.to(torch.float32)

    data_x = torch.cat((data_x,composition_mat_add), dim=1)
    data_y = data_y.view(-1,1)

    data_x = data_x[0:1000,:]
    data_y = data_y[0:1000,:]

    print(data_x.shape)
    print(data_y.shape)

    print(data_x[0])

    return data_x, data_y