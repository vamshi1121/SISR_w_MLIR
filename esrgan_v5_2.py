import argparse
import os
import numpy as np
import math
import itertools
import sys
import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
from utils import *

from models_v5 import *


#os.environ["CUDA_VISIBLE_DEVICES"]="3,4,6"
os.makedirs("../training", exist_ok=True)
os.makedirs("../saved_models", exist_ok=True)
os.makedirs("../eval_", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=154, help="epoch to start training from")
parser.add_argument("--phase", type=int, default=1, help="1 or 2")
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="NIH Chest X-Ray", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=140, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=130, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)
if opt.phase ==1:
    from datasets_phase1 import *
else:
    from datasets_phase2 import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
generator = nn.DataParallel(generator).to(device)
discriminator = nn.DataParallel(discriminator).to(device)
feature_extractor = FeatureExtractor()
feature_extractor = nn.DataParallel(feature_extractor).to(device)
# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("../saved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load("../saved_models/discriminator_%d.pth" % opt.epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), amsgrad = True)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), amsgrad = True)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

if opt.phase==1:
    dataloader = DataLoader(
    ImageDataset("path_to_images(hr)", hr_shape = hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
else:
    dataloader = DataLoader(
    ImageDataset("path_to_images(hr)", hr_shape = hr_shape, csv_file="csv_file_with_BB_Boxes"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

from math import log10, sqrt
import cv2

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr,mse

def func1(genenerator,limit=35):
    psnr = []
    mse = []
    csv_path = "validation_imgs_list(csv or txt)"
    eval_imgs_path = "validation_images_folder_path"
    df = pd.read_csv(csv_path,header=None)
    imgs = df[0].values
    for count,i in tqdm(enumerate(imgs)):
        img_path = eval_imgs_path + i
        ori=Image.open(img_path).convert('RGB').resize((1024,1024))
        lr=ori.resize((256,256))
        lr=Variable(transform(lr)).to(device).unsqueeze(0)
        with torch.no_grad():
            sr_image = denormalize(generator(lr)).cpu()[0]
        save_image(sr_image,"../eval_/sr.png")
        sr_image = Image.open("../eval_/sr.png").convert('RGB')
        ori = np.array(ori)
        sr_image = np.array(sr_image)
        evl=PSNR(ori,sr_image)
        mse.append(float(evl[1]))
        psnr.append(float(evl[0]))
        if count>limit:
            break

    return np.mean(psnr),np.mean(mse)

df = pd.DataFrame(columns=["Epoch","PSNR","MSE","Losses"])

# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    amloss_G = AverageMeter()
    amloss_D = AverageMeter()
    amloss_content = AverageMeter()
    amloss_GAN = AverageMeter()
    amloss_pixel = AverageMeter()
    loop1 = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, imgs in loop1:

        batches_done = epoch * len(dataloader) + i

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor)).to(device)
        imgs_hr = Variable(imgs["hr"].type(Tensor)).to(device)
        if opt.phase == 2:
            imgs_lr2 = Variable(imgs["lr2"].type(Tensor)).to(device)
            imgs_hr2 = Variable(imgs["hr2"].type(Tensor)).to(device)
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.module.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.module.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)
        amloss_pixel.update(loss_pixel.item(), imgs["lr"].size(0))

        if batches_done < opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
        amloss_GAN.update(loss_GAN.item(), imgs["lr"].size(0))

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr).detach()
        loss_content = criterion_content(gen_features, real_features)
        amloss_content.update(loss_content.item(), imgs["lr"].size(0))
        if opt.phase == 2:
            # Part 2: For MSE_boundingbox
            gen_hr2 = generator(imgs_lr2)
            loss_pixel2 = criterion_pixel(gen_hr2, imgs_hr2)
            # Content loss
            gen_features2 = feature_extractor(gen_hr2)
            real_features2 = feature_extractor(imgs_hr2).detach()
            loss_content2 = criterion_content(gen_features2, real_features2)


        # Total generator loss

        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel
        if opt.phase == 2:
            loss_G2 = loss_content2 + 0.1 * loss_pixel2
            loss_G = loss_G + 0.2 * loss_G2
        # Total generator loss

        #loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel
        amloss_G.update(loss_G.item(), imgs["lr"].size(0))
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        amloss_D.update(loss_D.item(), imgs["lr"].size(0))

        loss_D.backward()
        optimizer_D.step()
        loop1.set_postfix(loss_G=amloss_G.avg, loss_D=amloss_D.avg, loss_content=amloss_content.avg, loss_pixel=amloss_pixel.avg, loss_GAN=amloss_GAN.avg)

        # --------------
        #  Log Progress
        # --------------

    psnr,mse = func1(generator)
    print("[Epoch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f] [PSNR: %f, MSE: %f]"
        % (epoch,
            opt.n_epochs,
            loss_D.item(),
            loss_G.item(),
            loss_content.item(),
            loss_GAN.item(),
            loss_pixel.item(),
            psnr,mse))
    data = [epoch,psnr,mse,[loss_D.item(),loss_G.item(),
            loss_content.item(),loss_GAN.item(),loss_pixel.item()]]
    df.loc[len(df.index)] = data
    torch.save(generator.state_dict(), "../saved_models/generator_%d.pth" % epoch)
    torch.save(discriminator.state_dict(), "../saved_models/discriminator_%d.pth" %epoch)
    df.to_csv("../out_data.csv")
print(df)



