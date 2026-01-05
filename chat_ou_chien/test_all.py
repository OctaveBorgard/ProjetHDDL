#%% Load librarys
import sys, os
project_abs_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_abs_dir)
from utils import (create_df,
                   CatDogBinary,
                   CatDogBreed,
                   CatDogSegmentation,
                   show_images)
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from deepinv.utils import plot
from torchvision.transforms import v2

# %%
import os
from torchvision.transforms import v2
import torch
from utils import CatDogSegmentation, create_df, show_images
from models import Unet_Segmenter
from training.train_utils import dice_loss
data_dir = "data"
df = create_df(cls_list_path=os.path.join(data_dir,"annotations/trainval.txt"),
                     image_path=os.path.join(data_dir, "images"),
                     segmentation_annot_path=os.path.join(data_dir, "annotations/trimaps"))

catdog_seg = CatDogSegmentation(df=df)
model = Unet_Segmenter()


transform = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomGrayscale(p=0.1),
    v2.GaussianNoise(),
    v2.ColorJitter(),
    v2.RandomCrop((224,224), pad_if_needed=True),
    v2.ToDtype(dtype=torch.float32, scale=True)
])

catdog_seg.transform = transform

sample = catdog_seg[0]
img, mask = sample
#%%

output = model(img.unsqueeze(0))

loss = dice_loss(output, mask.unsqueeze(0))


print("Dice Score:", loss.item())

# %%
show_images([img, mask],
            title=["Image", "Segmentation Mask"])

# %%
import deepinv as dinv

imgs, masks = catdog_seg[:4]
imgs = torch.stack(imgs)
masks = torch.stack(masks)

fig = dinv.utils.plot([imgs, masks],
                      titles=["Images", "Segmentation Masks"],
                      show=True)

# %%
import torch
from torch import randint
from torchmetrics.segmentation import MeanIoU
miou = MeanIoU(num_classes=3, input_format="index")
preds = randint(0, 3, (100, 128, 128), generator=torch.Generator().manual_seed(42))
target = randint(0, 3, (100, 128, 128), generator=torch.Generator().manual_seed(43))

print(miou(preds, target))

# %%
sample = randint(0,3,(1,224,224))
print(torch.nn.functional.one_hot(sample, num_classes=3).shape)

# %%
import torch
from training.train_utils import DiceLoss, DiceCELoss

dice_loss_fn = DiceLoss(reduction="mean", ignore_index=0, need_softmax=True)
cedice_loss_fn = DiceCELoss(ce_weight=0.5, dice_weight=0.5, reduction="mean", need_softmax=True)

targets = torch.randint(0,3,(10,1,224,224))
preds = torch.rand((10,3,224,224))*2

print("Dice Loss:", dice_loss_fn(preds, targets).item())
print("CE + Dice Loss:", cedice_loss_fn(preds, targets).item())

# %%
