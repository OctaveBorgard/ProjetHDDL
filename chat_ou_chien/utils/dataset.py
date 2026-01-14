#%%
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.tv_tensors import Image as TvImage
from torchvision.tv_tensors import Mask as TvMask
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import os
import warnings
#%%

def create_df(cls_list_path: str, image_path: str,  segmentation_annot_path:str = None):
    """
    Create a DataFrame that maps images to their corresponding annotation.

    This function assumes that:
        - Images are stored as `.jpg` files inside `image_path`.
        - Segmentation annotations are stored as `.xml` files inside `segmentation_annot_path`.
        - Each annotation file has the same base filename as its corresponding image.

    Parameters
    ----------
    cls_list_path: str
        Path to the .txt file that lists image and classfication label
    image_path : str
        Path to the directory containing image files (`.jpg`).
    segmentation_annot_path : str (optional)
        Path to the directory containing segmentation masks (`.png`).
    

    Returns
    -------
    pandas.DataFrame with following columns: `image_path`, `label`, `breed_id`, `segm_path`
    """
    filenames = []
    label_list = []
    breed_ids = []
    if segmentation_annot_path is not None:
        seg_annot_list = []
    with open(cls_list_path, "r") as f:
        for line in f:
            if not line or line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) != 4:
                continue

            filenames.append(os.path.join(image_path,parts[0]+".jpg"))
            breed_ids.append(int(parts[1]) - 1)
            label_list.append(int(parts[2]) - 1)
            if segmentation_annot_path is not None:
                seg_annot_list.append(os.path.join(segmentation_annot_path,parts[0]+".png"))

    if segmentation_annot_path is not None:
        return  pd.DataFrame({
                    "image_path": filenames,
                    "label": label_list,
                    "breed_id": breed_ids,
                    "segm_path": seg_annot_list,
                })   
    else:
        return  pd.DataFrame({
                    "image_path": filenames,
                    "label": label_list,
                    "breed_id": breed_ids,
                })   

class CatDogDataset(Dataset):
    breeds =['Abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound',
            'beagle', 'Bengal', 'Birman', 'Bombay', 'boxer', 'British_Shorthair',
            'chihuahua', 'Egyptian_Mau', 'english_cocker_spaniel', 'english_setter',
            'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond',
            'leonberger', 'Maine_Coon', 'miniature_pinscher', 'newfoundland', 'Persian',
            'pomeranian', 'pug', 'Ragdoll', 'Russian_Blue', 'saint_bernard', 'samoyed',
            'scottish_terrier', 'shiba_inu', 'Siamese', 'Sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier',
             'yorkshire_terrier']
    breeds_to_ids = {breed: i for i, breed in enumerate(breeds)}
    species = ['cat', 'dog']

    def __init__(self, df: pd.DataFrame, transform: torchvision.transforms, *arg, **kwarg):
        """
        Parameters
        ----------
        pandas.DataFrame with following columns: `image_path`, `species`, `breed_id`, `segm_path` (optional)

       """
        super().__init__(*arg, **kwarg)
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        if not isinstance(index, int):
            
            imgs = []
            labels = []
            indices = None
            if isinstance(index, slice):
                start = index.start or 0
                stop = index.stop or len(self.df)
                step = index.step or 1
                indices = range(start, stop, step)
            elif isinstance(index, (list, tuple, np.ndarray)):
                indices = index
            else:
                raise TypeError("Index must be integer, slice, list or tuple")
            
            for i in indices:
                img, label = self._get_single(i)
                imgs.append(img)
                labels.append(label)

            return imgs, labels
        
        else:
            return self._get_single(index)
    
    def _get_single(self, index):
        raise NotImplemented
    

class CatDogBinary(CatDogDataset):
    def __init__(self, df: pd.DataFrame, transform: torchvision.transforms=None, *arg, **kwarg):
        super().__init__(df, transform, *arg, **kwarg)
    
    def _get_single(self, index):
        row = self.df.iloc[index]  
        img_path = row["image_path"]
        img = decode_image(img_path)[:3]
        if self.transform is not None:
            img = self.transform(img)
        label = row['label']
        return img, torch.tensor(label)
    
class CatDogBreed(CatDogDataset):
    def __init__(self, df: pd.DataFrame, transform: torchvision.transforms=None, *arg, **kwarg):
        super().__init__(df, transform, *arg, **kwarg)
    
    
    def _get_single(self, index):
        row = self.df.iloc[index]  
        img_path = row["image_path"]
        img = decode_image(img_path)[:3]
        if self.transform is not None:
            img = self.transform(img)

        label = row['breed_id']
        return img, torch.tensor(label)
    
class CatDogSegmentation(CatDogDataset):
    def __init__(self, df: pd.DataFrame, transform: torchvision.transforms=None, *arg, **kwarg):
        super().__init__(df, transform, *arg, **kwarg)
    
    def _get_single(self, index):
        row = self.df.iloc[index]  
        img_path = row["image_path"]
        img = decode_image(img_path)[:3]
        img = TvImage(img)

        
        segm_path = row["segm_path"]
        segm = decode_image(segm_path)[:1] - 1
        segm = TvMask(segm)
        
        if self.transform is not None:
            # Apply transforms to both image and mask together to keep them aligned
            img, segm = self.transform(img, segm)

        return img, segm
    
#



    

