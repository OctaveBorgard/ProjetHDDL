#%% Load librarys
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
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#%% Load images and Annotations
data_dir = os.path.join(project_abs_dir, "data")

df_train = create_df(cls_list_path=os.path.join(data_dir,"annotations/trainval.txt"),
                     image_path=os.path.join(data_dir, "images"),
                     segmentation_annot_path=os.path.join(data_dir, "annotations/trimaps"))

df_test = create_df(cls_list_path=os.path.join(data_dir,"annotations/test.txt"),
                    image_path=os.path.join(data_dir, "images")
                    )

catdog_bin_train = CatDogBinary(df_train)
catdog_bin_test = CatDogBinary(df_test)

catdog_breed_train = CatDogBreed(df_train)
catdog_breed_test = CatDogBreed(df_test)

catdog_segm = CatDogSegmentation(df_train)

species = CatDogBinary.species
breeds = CatDogBinary.breeds
#%% Visualize some samples images
indices = np.random.randint(low=0, high=1000, size=(10,))
img_samples, species_samples = catdog_bin_train[indices]
_, breed_samples = catdog_breed_train[indices]
annot = [f"{species[sp]} : {breeds[br]}" for sp, br in zip(species_samples, breed_samples)]
show_images(img_samples, title=annot, ncols=5)
# Personally, I do not see any visual biases, but maybe you guys can see some :).
# %% Distribution by species (dog or cat)
species_train = df_train[['label']]
species_train['animal'] = species_train['label'].map({0: 'cat', 1:'dog'})
species_train['source'] = "train"

species_test = df_test[['label']]
species_test['animal'] = species_test['label'].map({0: 'cat', 1:'dog'})
species_test['source'] = "test"

species_full = pd.concat([species_train, species_test])

sns.histplot(data=species_full, x="animal", hue="source",
             multiple="dodge",)
plt.title("Distribution of cat and dog on train set and test test ")
plt.show()

# As shown by the figure, the distribution of cats and dogs in both
# train set and test set is very unbalanced, the number of cats is roughly
# on half the number of dog. A positive point here is that the train set and
# test set come from a same distribution. 

# %% Distribution of breed
breed_train = df_train[['breed_id']]
breed_train['breed'] = breed_train['breed_id'].map(
    {i:breed for i, breed in enumerate(breeds)}
    )

breed_test = df_test[['breed_id']]
breed_test['breed'] = breed_test['breed_id'].map(
    {i:breed for i, breed in enumerate(breeds)}
    )

fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16,8))
sns.histplot(data=breed_train, x="breed", ax=axs[0])
axs[0].tick_params(axis='x', rotation=90)
axs[0].set_title("Distribution of breed in train set")

sns.histplot(data=breed_test, x="breed", ax=axs[1])
axs[1].tick_params(axis='x', rotation=90)
axs[1].set_title("Distribution of breed in test set")

plt.tight_layout()
plt.show()

# The distribution of breed is rather balance in both train set and test set
# around 100 individuals/images for each class, this quantity may be not enough for 
# breed classification task, we may consider the data augmentation for the training
# processus, for binary classfication task, data augmentation may not be required.


#%% show images along with their masks for segmentation task
n = 5
indices = np.random.randint(low=0, high=1000, size=(n,))
images_samples, mask_samples = catdog_segm[indices]
for i in range(n):
    show_images([images_samples[i], mask_samples[i]], title=["image", "segmentation mask"])

# The masks look coherent with the images, however we noticed a small inconsitence
# in the band width of the border lines (i.e non-classified region) as certain masks
# have thinner border than others.
# %%
