#%%
import sys, os
project_abs_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..","..")
)
sys.path.append(project_abs_dir)
from utils import (create_df,
                   CatDogBinary,
                   CatDogBreed,
                   CatDogSegmentation,
                   show_images)
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
from torchvision.models import efficientnet_b0

# %%
