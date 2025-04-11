# Copyright 2023 NNAISENSE SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import pathlib
import pickle
import zipfile
from typing import Union
import pandas as pd
import re
from collections import Counter

import numpy as np
import requests
import torch
import torchvision
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.utils import make_grid

from utils_model import quantize
from cadlib.cad_dataset import CADDataset
from cadlib.guidance_dataset import GuidanceDataset

TEXT8_CHARS = list("_abcdefghijklmnopqrstuvwxyz")


def bin_mnist_transform(x):
    return torch.bernoulli(x.permute(1, 2, 0).contiguous()).int()


def bin_mnist_cts_transform(x):
    return torch.bernoulli(x.permute(1, 2, 0).contiguous()) - 0.5


def rgb_image_transform(x, num_bins=256):
    return quantize((x * 2) - 1, num_bins).permute(1, 2, 0).contiguous()


class MyLambda(torchvision.transforms.Lambda):
    def __init__(self, lambd, arg1):
        super().__init__(lambd)
        self.arg1 = arg1

    def __call__(self, x):
        return self.lambd(x, self.arg1)


def make_datasets(cfg: DictConfig) -> tuple[Dataset, Dataset, Dataset]:
    """
    Mandatory keys: dataset (must be cifar10, mnist, bin_mnist, bin_mnist_cts or text8), data_dir
    Mandatory for text: seq_len
    """
    if cfg.dataset == "cad":
        train_set = CADDataset("train", cfg)
        val_set = CADDataset("validation", cfg)
        test_set = CADDataset("test", cfg)
    elif cfg.dataset == "cad_guidance":
        train_set = GuidanceDataset("train", cfg)
        val_set = GuidanceDataset("validation", cfg)
        test_set = GuidanceDataset("test", cfg)
    else:
        raise NotImplementedError(cfg.dataset)

    return train_set, val_set, test_set