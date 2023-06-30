import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.utils.data import DataLoader

import pandas as pd

class Dataset(data.Dataset):
    def __init__(self, data, file_names, labels, target_image_w=299,target_image_h=299,transform=True):
        self.root = data
        self.ids = file_names
        self.labels = labels
        self.target_image_w = target_image_w
        self.target_image_h = target_image_h

        # TODO: Apply different transformations for different datasets.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the tensor
            transforms.Resize((self.target_image_w, self.target_image_h))]
        )


    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id, label = self.ids[idx], self.labels[idx]
        file_name = f"{img_id}"
        path = osp.join(self.root, file_name)
        img = cv2.imread(path)
        img = self.transform(img)

        return (img, torch.tensor(label))