import glob
import random
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape,csv_file):
        hr_height, hr_width = hr_shape
        self.csv_file=pd.read_csv(csv_file)
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = list(root+"/"+self.csv_file["Image Index"])
        self.file_names = list(self.csv_file["Image Index"])
        self.csv_file=self.csv_file

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert('RGB')
        x,y,w,h = self.csv_file.loc[self.csv_file["Image Index"] == self.file_names[index % len(self.files)]][['Bbox [x', 'y', 'w', 'h]']].values[0]
        img2 = img.crop((int(x),int(y),int(x+w),int(y+h)))
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        img2_lr = self.lr_transform(img2)
        img2_hr = self.hr_transform(img2)

        return {"lr": img_lr, "hr": img_hr, "lr2": img2_lr, "hr2": img2_hr}

    def __len__(self):
        return len(self.files)

