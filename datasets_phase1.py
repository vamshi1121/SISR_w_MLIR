import glob
import random
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
def read_xray(path, voi_lut = False, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)   
    return data

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape,path = "/DATA2/VinDr-CXR/train/"):
        hr_height, hr_width = hr_shape
        #df=pd.read_csv(csv_file, header=None)
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
        self.path = path
        self.files = os.listdir(path)
        #self.file_names = list(df[0])
        #self.csv_file=df

    def __getitem__(self, index):
        img = self.files[index]
        img = read_xray(self.path+img)
        img = Image.fromarray(img).convert('RGB').resize((1024,1024))
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)
