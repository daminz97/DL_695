import argparse
import torch
import os
import numpy as np
import glob
import random
import torchvision.transforms as tvt

from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageDataloader(Dataset):
    def __init__(self, data_path, class_list, transforms=None):
        self.label_dict = {'airplane':0,
                           'boat':1,
                           'cat':2,
                           'dog':3,
                           'elephant':4,
                           'giraffe':5,
                           'horse':6,
                           'refrigerator':7,
                           'train':8,
                           'truck':9}
        self.img_path = []
        self.labels = []
        for path in class_list:
            folder_path = os.path.join(data_path, path+'/*')
            for name in glob.glob(folder_path):
                self.img_path.append(name)
                self.labels.append(self.label_dict[path])
        self.transforms = transforms

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        if self.transforms is not None:
            img_norm = self.transforms(img)
        label = self.labels[index]

        return img_norm, label