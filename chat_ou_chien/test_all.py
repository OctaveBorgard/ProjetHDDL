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
#%% Load images and Annotations
data_dir = os.path.join(project_abs_dir, "data")

df = create_df(cls_list_path=os.path.join(data_dir,"annotations/list.txt"),
                     image_path=os.path.join(data_dir, "images"))
# df_train = create_df(cls_list_path=os.path.join(data_dir,"annotations/trainval.txt"),
#                      image_path=os.path.join(data_dir, "images"),
#                      segmentation_annot_path=os.path.join(data_dir, "annotations/trimaps"))

# df_test = create_df(cls_list_path=os.path.join(data_dir,"annotations/test.txt"),
#                     image_path=os.path.join(data_dir, "images")
#                     )

catdog = CatDogBinary(df=df, transforms=v2.Resize((224,224)))


species = CatDogBinary.species
breeds = CatDogBinary.breeds

radio = 0.9
train_size = int(radio*len(catdog))
test_size = len(catdog) - train_size

train_set, test_set = random_split(catdog,[train_size, test_size])
#%%
samples, labels = train_set[:10]
show_images(samples,
     title=[catdog.species[label] for label in labels])
# %%
for i in range(len(catdog)):
    image,_ = catdog[i]
    if image.shape[0] != 3:
        break
print(i)
# %%
from PIL import Image
path = df.loc[i,"image_path"]
print(path)
img_pil = Image.open(path)
img_pil
# %%
img_torch = torchvision.io.decode_image(path)
show_images([img_torch[:3], img_torch[-3:],img_torch])

# %%
