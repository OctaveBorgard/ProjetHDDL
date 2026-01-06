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
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from deepinv.utils import plot
from torch.utils.data import random_split, RandomSampler, DataLoader
from training import (collate_fn,
                      LoggingConfig,
                      TrainingConfig)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
train_test_ratio = 0.9
batch_size = 64
save_dir = "exp/fine_classification"

#################### DEFINE DATASET ##############################
df = create_df(cls_list_path=os.path.join(project_abs_dir, "data/annotations/list.txt"),
                image_path=os.path.join(project_abs_dir, "data/images"))
dataset_full = CatDogBreed(df)
class_str = dataset_full.breeds

train_size = int(train_test_ratio*len(dataset_full))
test_size = len(dataset_full) - train_size
print(f"Train set size: {train_size}, test set size: {test_size}")

g = torch.Generator().manual_seed(42)
train_dataset, test_dataset = random_split(dataset_full,
                                            [train_size, test_size],
                                            generator=g)

transform_eval =  v2.Compose([
    v2.Resize((224,224)),
    v2.ToDtype(dtype=torch.float32, scale=True)
])

train_dataset.dataset.transform = transform_eval
test_dataset.dataset.transform = transform_eval 

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                shuffle=False, pin_memory=True, drop_last=False,
                                num_workers=8)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
#%%
########################### DEFINE MODEL ########################
model_kwargs = dict(weights=EfficientNet_B0_Weights.DEFAULT)
model = efficientnet_b0(**model_kwargs)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_str))



######################### training logger ##################

logger = LoggingConfig(project_dir=os.path.join(project_abs_dir, save_dir),
                        exp_name=f"EfficientNet_{train_size}")
logger.monitor_metric = "test_avg_loss"
logger.monitor_mode = "min"

state = logger.load_latest_checkpoint()
                                    
# state = logger.load_best_checkpoint()
model.load_state_dict(state['model_state_dict'])
model.to(device)
model.eval()

#%%

train_accuracy = 0
test_accuracy = 0


cnt = 0
mapping = np.zeros(shape=(len(class_str),))
breed_images = dict()
for img, label in dataset_full:
    if mapping[label] == 0:
        breed_images[label.item()] = img
        mapping[label] = 1
        cnt += 1
        if cnt == len(class_str):
            break

train_error_df = dict(gt_breed=[], pred_breed=[], gt_index=[], pred_index=[])
test_error_df = dict(gt_breed=[], pred_breed=[], gt_index=[], pred_index=[])

with torch.no_grad():
    for images, labels in train_loader:
        images = torch.stack(images).to(device)
        labels = torch.tensor(labels, device=device)

        pred_labels = model(images).detach().argmax(
            dim=1, keepdim=False)
        indice_fail = (labels != pred_labels)
        
        train_error_df["gt_breed"].extend([class_str[label] for label in labels[indice_fail].tolist()])
        train_error_df["pred_breed"].extend([class_str[pred_label] for pred_label in pred_labels[indice_fail].tolist()])
        train_error_df["gt_index"].extend([label for label in labels[indice_fail].tolist()])
        train_error_df["pred_index"].extend([pred_label for pred_label in pred_labels[indice_fail].tolist()])
        if torch.sum(indice_fail) > 0:
            show_images(images[indice_fail], title=[f"T:{labels[i].item()}, P:{pred_labels[i].item()}" for i in torch.where(indice_fail)[0]])
        train_accuracy += torch.sum(labels == pred_labels).item()
    train_accuracy = train_accuracy/train_size

    for images, labels in val_loader:
        images = torch.stack(images).to(device)
        labels = torch.tensor(labels, device=device)

        pred_labels = model(images).detach().argmax(
            dim=1, keepdim=False)
        
        indice_fail = labels != pred_labels

        test_error_df["gt_breed"].extend([class_str[label] for label in labels[indice_fail].tolist()])
        test_error_df["pred_breed"].extend([class_str[pred_label] for pred_label in pred_labels[indice_fail].tolist()])
        test_error_df["gt_index"].extend([label for label in labels[indice_fail].tolist()])
        test_error_df["pred_index"].extend([pred_label for pred_label in pred_labels[indice_fail].tolist()])

        if torch.sum(indice_fail) > 0:
            show_images(images[indice_fail], title=[f"T:{labels[i].item()}, P:{pred_labels[i].item()}" for i in torch.where(indice_fail)[0]])

        test_accuracy += torch.sum(labels == pred_labels).item()
    test_accuracy = test_accuracy/test_size

print(f"Accuracy obtained on train set: {train_accuracy:.3f}, and on test set: {test_accuracy:.3f}")
    



# %%
### Suspection confusing races 

import pandas as pd

train_error_pdf = pd.DataFrame(train_error_df)
train_error_pdf = train_error_pdf.value_counts().reset_index(name='Count')
test_error_pdf = pd.DataFrame(test_error_df)
test_error_pdf = test_error_pdf.value_counts().reset_index(name='Count')
# %%
print("############### Confusion in train set #####################")
for i in range(len(train_error_pdf)):
    row = train_error_pdf.iloc[i]
    gt_breed = row['gt_breed']
    pred_breed = row['pred_breed']
    gt_index = row['gt_index']
    pred_index = row['pred_index']
    Count = row['Count']

    print(f"The model is wrong {Count} times, it predict a {gt_breed}:{gt_index} as a {pred_breed}:{pred_index}")
    show_images([breed_images[gt_index], breed_images[pred_index]], title=[str(gt_index), str(pred_index)])

    

# %%
print("############### Confusion in test set #####################")
for i in range(len(test_error_pdf)):
    row = test_error_pdf.iloc[i]
    gt_breed = row['gt_breed']
    pred_breed = row['pred_breed']
    gt_index = row['gt_index']
    pred_index = row['pred_index']
    Count = row['Count']

    print(f"The model is wrong {Count} times, it predict a {gt_breed}:{gt_index} as a {pred_breed}:{pred_index}")
    show_images([breed_images[gt_index], breed_images[pred_index]], title=[str(gt_index), str(pred_index)])
# %%
transform_train = v2.Compose([
    v2.Resize((224,224)),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.RandomHorizontalFlip(),
    v2.RandomGrayscale(p=0.1),
    v2.ColorJitter(),
    v2.GaussianNoise(),
    v2.RandomRotation(degrees=15),
])



# %%
dataset_full_raw = CatDogBreed(df)
image_size = set([img.shape[-2:] for img, _ in dataset_full_raw])

# %%
image_size = list(image_size)
print(image_size[:4])
# %%
min_h = min([size[0] for size in image_size])
min_w = min([size[1] for size in image_size])
print(f"Minimum height: {min_h}, minimum width: {min_w}")

# %%
